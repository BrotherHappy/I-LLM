from collections import OrderedDict
from quantize.quant_modules import QuantLinear
import torch
from quantize.quant_modules import QuantMatMul
from models.transformation import *


def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def rescale_paramters(model):
    params = []
    template = "rescale_param"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def get_fsbr_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1 or n.find("rescale") > -1:
            params.append(m)
    return iter(params)  

def fsbr_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1 or name.find("rescale") > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args, isllama,quant_temp=True): # don't modify quant_temp
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj], # 利用前面的smooth系数来smooth qkv_linear
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj], # 利用post_norm来smooth up_proj和gate_proj的输入激活
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj, # 利用v_proj的结果来smooth out_proj的输入激活
                                model.out_smooth_scale, model.out_smooth_shift,self_attn=model.self_attn)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj, # 平衡q和k两者的输出激活
                                model.qkt_smooth_scale,self_attn=model.self_attn)
            if hasattr(model,"x1x2_smooth_scale"):# 用来进行x1*x2*sidmoid(x2)的smooth
                smooth_q_k_temporary(model.mlp.up_proj,model.mlp.gate_proj,model.x1x2_smooth_scale)
                model.mlp.swiglu.smooth = model.x1x2_smooth_scale
            smooth_fc_fc_temporary(model.mlp.up_proj,model.mlp.down_proj,model.fc2_smooth_scale,None) # 进行up & down的平衡
            # model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else: # for opt
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            # smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
            #                     model.out_smooth_scale, model.out_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale,model.out_smooth_shift) # v和o不应该增加shift,因为有非线性激活函数
            smooth_fc_fc_temporary(model.fc1,model.fc2,model.fc2_smooth_scale)
        
            # model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    if quant_temp:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args, isllama):
    if args.let:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift,self_attn=model.self_attn)
            smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale,self_attn = model.self_attn)
            if hasattr(model,"x1x2_smooth_scale"):
                smooth_q_k_inplace(model.mlp.up_proj,model.mlp.gate_proj,model.x1x2_smooth_scale,self_attn=model.self_attn)
                model.mlp.swiglu.smooth = model.x1x2_smooth_scale
            smooth_fc_fc_inplace(model.mlp.up_proj,model.mlp.down_proj,model.fc2_smooth_scale,None) # 进行up & down的平衡
        else: # opt
            smooth_ln_fcs_inplace(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_inplace(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale,self_attn = model.self_attn)
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale,model.out_smooth_shift)
            smooth_fc_fc_inplace(model.fc1,model.fc2,
                                model.fc2_smooth_scale)
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)


import tqdm
import os
import jsonlines

"""
git clone https://github.com/openai/human-eval
$ pip install -e human-eval
evaluate_functional_correctness sample-output-file
"""
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer

def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.decode(
            tokens[raw_text_len:])
        sent = sent.split("def ")[0]
        sent = sent.split('\n\n\n')[0]
        sent = sent.split("</s>")[0]
        sent = "    " + sent.lstrip()
        sents.append(sent)
    return sents

def generate_sample(model, tokenizer, input_txt):
    #print(f"Input text: {input_txt}\n")
    inputs = tokenizer(input_txt, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
    raw_text_len = len(inputs["input_ids"][0])
    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    output_text = decode(outputs,tokenizer,raw_text_len)[0]
    #print(f"\nOutput text: \n{output_text}\n")
    return output_text

def generate_sample_instruct(model, tokenizer, input_txt):
    #print(f"Input text: {input_txt}\n")
    messages = [
        {"role": "system", "content": "You are a expert in the field of computer science. You are asked to complete the code based on the following prompt.ONLY give the code."},
        {"role": "user", "content": input_txt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        # return_dict=True, # 3.35 not support return_dict=True
    ).to(model.device)

    # terminators = [
    #     tokenizer.eos_token_id,
    #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]
    # inputs = tokenizer(input_txt, return_tensors="pt", truncation=True, max_length=2048).to('cuda')
    # 3.35版本的不可以返回dict
    # raw_text_len = len(inputs["input_ids"][0])
    # outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
    raw_text_len = len(inputs) # 
    outputs = model.generate(inputs,do_sample=False,max_new_tokens=512)
    output_text = decode(outputs,tokenizer,raw_text_len)[0]
    #print(f"\nOutput text: \n{output_text}\n")
    return output_text

def eval_humaneval(model,tokenizer,logpath):
    f_output = jsonlines.Writer(open(os.path.join(logpath,"HumanEval_res.jsonl"), 'w', encoding='utf-8'))

    f = jsonlines.open("data/human-eval/data/HumanEval.jsonl")
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc='task_idx'):
            prompt = jobj['prompt']
            task_id = jobj['task_id']
            gen_sents = generate_sample(model, tokenizer, prompt)
            gen_jobjs = {'task_id': task_id, "completion": gen_sents.replace("<|end_of_text|>","")} 
            output.write(gen_jobjs)
    f_output.close()

    # os.system("evaluate_functional_correctness HumanEval_res.jsonl")

    return 0


def eval_humaneval_instruct(model,tokenizer,logpath="logs",logger=None):
    logpath = os.path.join(logpath,"human_eval")
    os.makedirs(logpath,exist_ok=True)
    f_output_path = os.path.join(logpath,"HumanEval_res.jsonl")
    f_output = jsonlines.Writer(open(f_output_path, 'w', encoding='utf-8'))
    log_fn = print if logger is None else logger.info

    f = jsonlines.open("data/human-eval/data/HumanEval.jsonl")
    with f_output as output:
        for jobj in tqdm.tqdm(f, desc='task_idx'):
            prompt = jobj['prompt']
            task_id = jobj['task_id']
            gen_sents = generate_sample_instruct(model, tokenizer, prompt)
            gen_jobjs = {'task_id': task_id, "completion": gen_sents.replace("<|end_of_text|>","")} 
            output.write(gen_jobjs)
    f_output.close()
    res = os.system(f"pip install human_eval && evaluate_functional_correctness {f_output}")
    if res!=0:
        log_fn("humean error")
    return res
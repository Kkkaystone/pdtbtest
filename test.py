

# str="111\n||||||\n1111"

# print(str.split('||||||'))
#auth_token="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"
#read token
auth_token="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"
from tqdm import tqdm
import os
import torch
from peft import LoraConfig, PeftModel
from implictdatareader import load_pdtb,transform_train_conversation,transform_test_conversation

from huggingface_hub import HfApi, Repository
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
device_map = {"": 0}

model_name= "meta-llama/Llama-2-7b-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=True)
PATH=new_model="/scratch/user/xishi/pdtb/coding/llama-2-7b-pdtb2.0-epoch3"

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
    use_auth_token=auth_token
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=auth_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# load test set
test_dataset=load_pdtb(split="test")
prompts=test_dataset.map(transform_test_conversation)

batch_size=8
out_list = []


# model.eval()

# with torch.no_grad():
#     batch_prompts = prompts['text'][0]
#     print(batch_prompts)
#     model_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to('cuda')
#     model = model.bfloat16().cuda()
#     outputs = model.generate(
#             **model_inputs,
#             max_new_tokens=8,  # 根据需要调整
#         )
#     out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     print(out_sentence)
    # 使用tqdm显示进度
    # for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
    #     batch_prompts = prompts['text'][i:i+batch_size]
    #     model_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to('cuda')
        
    #     outputs = model.generate(
    #         **model_inputs,
    #         max_new_tokens=8,  # 根据需要调整
    #     )
    #     out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #     out_list += out_sentence
    # print(out_list)
    # pred = out_list.split("[/INST]")[1]
    
    # pred = pred.split(' ')



    # print(len(pred))
    # print()
    # print(type(pred))
    # print()
    # print(len(pred))

# del pipe
# # Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=auth_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
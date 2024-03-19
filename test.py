

# str="111\n||||||\n1111"

# print(str.split('||||||'))
#auth_token="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"
#read token
auth_token="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"
from tqdm import tqdm
import json
import os
import torch
from datasets import load_dataset,load_from_disk
from peft import LoraConfig, PeftModel
from implictdatareader import load_pdtb,transform_train_conversation,transform_test_conversation,transform_test_conversation_fewshot
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import HfApi, Repository
from datetime import datetime
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

PATH=new_model="/scratch/user/xishi/pdtb/coding/llama-2-7b-pdtb2.0-epoch3-p420240317180423"

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
    repetition_penalty=1.5,
    use_auth_token=auth_token
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# # Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=auth_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# load test set
test_dataset=load_pdtb(split="test")

prompts=test_dataset.map(transform_test_conversation)


fewshot_dataset="fewshot_dataset_p4_fix"
try:
    prompts_fewshot = load_from_disk(fewshot_dataset)
except Exception as e:
    print(f"An error occurred: {e}")
    print("fewshot_dataset not found, mapping from test_dataset")
    prompts_fewshot=test_dataset.map(transform_test_conversation_fewshot)
try:
    prompts_fewshot.save_to_disk("fewshot_dataset")
except PermissionError:
    print(f"Tried to overwrite but a dataset can't overwrite itself.")



batch_size=1

category_mapping = {'Temporal': 0, 'Comparison': 1, 'Contingency': 2, 'Expansion': 3}




label_true=[]
for data in test_dataset:
    label=[0,0,0,0]
    for category in category_mapping:
        if category in data['answer']:
            label[category_mapping[category]]=1    
    label_true.append(label)
def label_mapping(out_list,is_sft=False):
    labels=[]
    cnt=0
    for out_sentence in out_list:
        answer=out_sentence.split("### Response:")[-1]
        label=[0,0,0,0]
        for category in category_mapping:
            if category in answer:
                label[category_mapping[category]]=1    
        if all(x==0 for x in label):
            cnt+=1
            print(out_sentence)
            if is_sft: 
                answer=out_sentence.split("### Response:")[-2]
            label=[0,0,0,0]
            for category in category_mapping:
                if category in answer:
                    label[category_mapping[category]]=1  
        labels.append(label)
    print("len(out_list)",len(out_list))
    print("all_0_count:",cnt)
    return labels

def eval(model,prompts,test_type):
    model.eval()
    out_list=[]
    with torch.no_grad():
        # batch_prompts = prompts['text'][0]
        # print(batch_prompts)
        # model_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to('cuda')
        # model = model.bfloat16().cuda()
        # outputs = model.generate(
        #         **model_inputs,
        #         max_new_tokens=8
        #     )
        # out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print("out_sentence",out_sentence)
        # print("true label:",test_dataset['answer'][0])
        # ###使用tqdm显示进度
        
        # for i in tqdm(range(0, len(prompts), batch_size), desc=f"{test_type} Processing batches"):
        #     batch_prompts = prompts['text'][i:i+batch_size]
        for i in tqdm(range(0, len(prompts)), desc=f"{test_type} Processing "):
            batch_prompts = prompts['text'][i]
            model_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to('cuda')
            model = model.bfloat16().cuda()
            outputs = model.generate(
                **model_inputs,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                max_new_tokens=10
            )
            out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            out_list += out_sentence
        with open('../output/'+'output_'+test_type+datetime.now().strftime("%m%d%H%M%S")+".json", 'w', encoding='utf-8') as file:
            # 将列表转换为JSON格式的字符串并写入文件
            json.dump(out_list, file)
            print("successfully save output file")
    return out_list
out_list_sft = eval(model,prompts,"sft")
out_list_zeroshot=eval(base_model,prompts,"zeroshot")
out_list_fewshot=eval(base_model,prompts_fewshot,"fewshot")
# with open('../output/output_zeroshot0318124356.json', 'r', encoding='utf-8') as file:
#     out_list_zeroshot = json.load(file)

# with open('../output/output_fewshot0318125230.json', 'r', encoding='utf-8') as file:
#     out_list_fewshot= json.load(file)
# with open('../output/output_sft0318170445.json', 'r', encoding='utf-8') as file:
#     out_list_sft= json.load(file)


label_predict_zeroshot=label_mapping(out_list_zeroshot)
label_predict_fewshot=label_mapping(out_list_fewshot)
label_predict_sft=label_mapping(out_list_sft,True)

label_true=[]
for data in test_dataset:
    label=[0,0,0,0]
    for category in category_mapping:
        if category in data['answer']:
            label[category_mapping[category]]=1    
    label_true.append(label)

def print_eval_result(label_predict,test_type):
    f1_macro = f1_score(label_true, label_predict, average='macro')
    print("f1_macro"+"_"+test_type,f1_macro)
    f1_micro = f1_score(label_true, label_predict, average='micro')
    print("f1_micro"+"_"+test_type,f1_micro)
    f1_weighted = f1_score(label_true, label_predict, average='weighted')
    print("f1_weighted"+"_"+test_type,f1_weighted)
    f1_none = f1_score(label_true, label_predict, average=None)
    print("f1_none"+"_"+test_type,f1_none)
    print("accuracy_score"+'_'+test_type,accuracy_score(label_true, label_predict))


print_eval_result(label_predict_zeroshot,"zeroshot")
print_eval_result(label_predict_fewshot,"fewshot")
print_eval_result(label_predict_sft,"sft")




# # del pipe
# # # Reload tokenizer to save it
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=auth_token)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"



# model.push_to_hub(new_model, use_temp_dir=False)
# tokenizer.push_to_hub(new_model, use_temp_dir=False)
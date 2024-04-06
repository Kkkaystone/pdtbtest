

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
device_map = 'auto'

model_name= "meta-llama/Llama-2-70b-chat-hf"
print("####using model:", model_name)

PATH=new_model="/scratch/user/xishi/pdtb/coding/llama-2-7b-pdtb2.0-epoch3-p420240317180423"
# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
# ## Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map=device_map,
    repetition_penalty=1.5,
    attn_implementation="flash_attention_2",
    use_auth_token=auth_token
)
# # print("####base_model",base_model)
# model = PeftModel.from_pretrained(base_model, new_model)
# model = model.merge_and_unload()
# print("####base_model",base_model)

# # Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=auth_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


#fewshot prompt:
prompts_fewshot_template=f'''
### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>Was this why some of the audience departed before or during the second half?
Or was it because Ms. Collins had gone</ARG1><ARG2>Either way it was a pity</ARG2>

### Response:
Let's think step by step, and give a summary at the end:

Identifying the Arguments:

ARG1 questions the reason for the audience leaving during a performance.
ARG2 comments on the situation, noting it as unfortunate regardless of the reason.
Determining the Relationship:

ARG2 extends the scenario presented in ARG1 by providing an evaluative remark on the outcome.
Reason for Expansion:

Without specifying or resolving the reasons posed in ARG1, ARG2 broadens the discussion by adding a perspective on the overall situation.
Summary: The classification as "Expansion" is because ARG2 broadens the discussion initiated in ARG1 with an evaluative statement, adding to the context without altering its course.

Expansion

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>The rest of the concert was more straight jazz and mellow sounds written by Charlie Parker, Ornette Coleman, Bill Douglas and Eddie Gomez, with pictures for the Douglas pieces</ARG1><ARG2>It was enjoyable to hear accomplished jazz without having to sit in a smoke-filled club, but like the first half, much of it was easy to take and ultimately forgettable</ARG2>

### Response:
Let's think step by step, and give a summary at the end:

Identifying the Arguments:

ARG1 describes the music played in the second half of a concert.
ARG2 expresses enjoyment of the music but notes it was ultimately forgettable.
Determining the Relationship:

ARG2 extends ARG1 by evaluating the concert experience, without contradicting or providing a causal link.
Reason for Expansion:

ARG2 adds to the description in ARG1 by sharing a personal reaction, fitting the "Expansion" category.
Summary: The classification as "Expansion" is due to ARG2 adding an evaluative perspective to the concert's description in ARG1, enhancing the information without altering its course or presenting an opposing viewpoint.

Expansion

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>Mr. Mayor's hope that references to "press freedom" would survive unamended seems doomed to failure</ARG1><ARG2>the current phrasing is "educating the public and media to avoid manipulation</ARG2>

### Response:
Let's think step by step, and give a summary at the end:
Identifying the Arguments:

ARG1 expresses concern about changes to "press freedom" references.
ARG2 presents the current phrasing regarding media education.
Determining the Relationship:

ARG2 specifies the nature of the amendments feared in ARG1, showing a cause-and-effect link.
Reason for Contingency:

The relationship is "Contingency" because ARG2 illustrates the specific outcome or effect (the amendment details) of the situation mentioned in ARG1 (concern about preserving "press freedom").
Summary: The classification as "Contingency" comes from ARG2 demonstrating the specific change or consequence that ARG1 was concerned about, indicating a cause-and-effect relationship between the two arguments.

Contingency

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>The Orwellian "New World Information Order" would give government officials rights against the press</ARG1><ARG2>journalists would be obliged to kowtow to their government, which would have licensing and censorship powers and, indeed, duties to block printing of "wrong" ideas</ARG2>

### Response:
Let's think step by step, and give a summary at the end:

Identifying the Arguments:

ARG1 mentions a proposed "New World Information Order" affecting press freedom.
ARG2 describes consequences for journalists under this order.
Determining the Relationship:

ARG2 provides specific effects of the "New World Information Order" mentioned in ARG1.
Reason for Contingency:

ARG2 shows the direct outcomes of the situation in ARG1, indicating a cause-and-effect link.
Summary: The classification as "Contingency" is due to ARG2 illustrating the direct consequences for journalists of the "New World Information Order" introduced in ARG1.

Contingency

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>He will be in charge of research, equity sales and trading, and the syndicate operation of Rothschild</ARG1><ARG2>Mr. Conlon was executive vice president and director of the equity division of the international division of Nikko Securities Co</ARG2>

### Response:
Let's think step by step, and give a summary at the end:

Identifying the Arguments:

ARG1 states a new responsibility for an individual at Rothschild.
ARG2 provides background on Mr. Conlon's previous position at Nikko Securities Co.
Determining the Relationship:

ARG2 describes Mr. Conlon's past role, which is logically prior to the new role mentioned in ARG1.
Reason for Temporal:

The sequence implies a before-and-after relationship, with ARG2 setting a temporal context for ARG1.
Summary: The classification as "Temporal" is due to ARG2 providing a background context that precedes the new appointment described in ARG1, indicating a time-based sequence of events.

Temporal

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>He gave up seven hits, walked five and didn't get a decision</ARG1><ARG2>Arm troubles forced him back to the minors the next year</ARG2>

### Response:
Let's think step by step, and give a summary at the end:
Identifying the Arguments:

ARG1 describes a baseball player's performance in a game.
ARG2 explains the player's subsequent demotion to the minors due to arm troubles.
Determining the Relationship:

ARG2 follows ARG1 in time, linking a poor performance to later consequences for the player.
Reason for Temporal:

The events in ARG2 (demotion due to arm troubles) occur after the events in ARG1 (game performance), suggesting a sequence.
Summary: The "Temporal" classification reflects ARG2 happening after ARG1, with the player's performance leading to future events affecting his career.

Temporal

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>Seats currently are quoted at $400,000 bid, $425,000 asked</ARG1><ARG2>The record price for a full membership on the exchange is $550,000, set March 9</ARG2>

### Response:
Let's think step by step, and give a summary at the end:
Identifying the Arguments:

ARG1 mentions the current bid and ask prices for seats on an exchange.
ARG2 states the record price for full membership on the same exchange.
Determining the Relationship:

ARG2 compares the current prices mentioned in ARG1 to a historical high.
Reason for Comparison:

The inclusion of a record price in ARG2 serves to compare or contrast with the current prices provided in ARG1.
Summary: The classification as "Comparison" is due to ARG2 presenting a historical price point that serves as a benchmark for the current prices mentioned in ARG1, facilitating a direct comparison.

Comparison

### Instruction:
Predict the relation between Arg1 and Arg2, just choose one or two label from :[Temporal, Comparison, Contingency, Expansion], no need to explain

### Input:
<ARG1>In late trading, the shares were up a whopping 122 pence ($1.93) -- a 16.3% gain -- to a record 869 pence on very heavy volume of 9.7 million shares</ARG1><ARG2>In the U.S. over-the-counter market, Jaguar shares trading as American Depositary Receipts closed at $13.625, up $1.75</ARG2>

### Response:
Let's think step by step, and give a summary at the end:
Identifying the Arguments:

ARG1 details a significant increase in share price on one market.
ARG2 describes the performance of the same shares in a different market.
Determining the Relationship:

Both ARG1 and ARG2 report on the financial performance of the same shares, but in different trading venues.
Reason for Comparison:

The juxtaposition of share price movements in two different markets naturally invites a comparison between the two scenarios.
Summary: The "Comparison" classification stems from ARG1 and ARG2 providing information on the performance of the same shares in two distinct markets, allowing for a comparative analysis of their respective price movements.
Comparison
'''


# load test set
test_dataset=load_pdtb(split="test")

prompts=test_dataset.map(transform_test_conversation)
prompts_cot=prompts.map(lambda x: {'text':x['text']+"Let's think step by step, and give a summary at the end:"})
prompts_cot_fewshot=prompts_cot.map(lambda x: {'text':prompts_fewshot_template+x['text']})

# fewshot_dataset="fewshot_dataset_p4_fix"
# try:
#     prompts_fewshot = load_from_disk(fewshot_dataset)
# except Exception as e:
#     print(f"An error occurred: {e}")
#     print("fewshot_dataset not found, mapping from test_dataset")
#     prompts_fewshot=test_dataset.map(transform_test_conversation_fewshot)
# try:
#     prompts_fewshot.save_to_disk("fewshot_dataset")
# except PermissionError:
#     print(f"Tried to overwrite but a dataset can't overwrite itself.")



batch_size=1

# category_mapping = {'Temporal': 0, 'Comparison': 1, 'Contingency': 2, 'Expansion': 3}




category_mapping = {'temporal': 0, 'comparison': 1, 'contingency': 2, 'expansion': 3}

label_true=[]
for data in test_dataset:
    label=[0,0,0,0]
    for category in category_mapping:
        if category in data['answer'].lower():
            label[category_mapping[category]]=1    
    label_true.append(label)
def label_mapping(out_list,is_fewshot=False):
    labels=[]
    cnt=0
    print(len(out_list))
    for i,out_sentence in enumerate(out_list):
        print(i)
        if is_fewshot:
            print("testest",i)
            print("".join(out_sentence.split("### Instruction:")[9:]))
            # print("out_sentence",out_sentence.split("### Response:")[8])
            # print("test_dataset",test_dataset['text'][i])
            answers=out_sentence.split("### Response:")[9:]
            answers=" ".join(answers).strip().lower()
        else:
            answers=out_sentence.split("### Response:")[1:]
            answers=" ".join(answers).strip().lower()

        label=[0,0,0,0]
        cnt,max_cnt,idx=0,0,-1
        print("###true_label",test_dataset['answer'][i])
        for category in category_mapping:
            cnt=answers.count(category)
            print(category,":",cnt)
            if max_cnt<cnt:
                max_cnt=cnt
                idx=category_mapping[category]
        if idx!=-1:
            label[idx]=1
            print('idx',idx)
        if all(x==0 for x in label):
            print(answers)
            cnt+=1    
        labels.append(label)
    print("len(out_list)",len(out_list))
    print("all_0_count:",cnt)
    print(labels)
    return labels

def eval(model,prompts,test_type):
    print("using model:", model)
    model.eval()
    out_list=[]
    with torch.no_grad():
        # batch_prompts = prompts['text'][0]
        # print(batch_prompts)
        # model_inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to('cuda')
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
            outputs = model.generate(
                **model_inputs,
                do_sample=True,
                temperature=0.75,
                top_p=0.9,
                max_new_tokens=1024
            )
            out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            out_list += out_sentence
        with open('../output/'+'output_'+test_type+datetime.now().strftime("%m%d%H%M%S")+".json", 'w', encoding='utf-8') as file:
            # 将列表转换为JSON格式的字符串并写入文件
            json.dump(out_list, file)
            print("successfully save output file")
    return out_list
# out_list_zeroshot=eval(base_model,prompts,"zeroshot")

# out_list_fewshot=eval(base_model,prompts_fewshot,"fewshot")
# out_list_zeroshot_cot=eval(base_model,prompts_cot,"zeroshot")
# out_list_sft = eval(model,prompts,"sft")

out_list_fewshot_cot=eval(base_model,prompts_cot_fewshot,"fewshot")
# with open('../output/output_zeroshot0328110333.json', 'r', encoding='utf-8') as file:
#     out_list_zeroshot = json.load(file)

# with open('../output/output_fewshot0318125230.json', 'r', encoding='utf-8') as file:
#     out_list_fewshot= json.load(file)
# with open('../output/output_fewshot0327162003.json', 'r', encoding='utf-8') as file:
#     out_list_fewshot_cot= json.load(file)
# with open('../output/output_sft0318170445.json', 'r', encoding='utf-8') as file:
#     out_list_sft= json.load(file)

# label_predict_zeroshot=label_mapping(out_list_zeroshot)

# label_predict_zeroshot_cot=label_mapping(out_list_zeroshot_cot)
label_predict_fewshot_cot=label_mapping(out_list_fewshot_cot,True)
# label_predict_sft=label_mapping(out_list_sft,True)



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

# print_eval_result(label_predict_zeroshot,"zeroshot")

# print_eval_result(label_predict_zeroshot_cot,"zeroshot")
print_eval_result(label_predict_fewshot_cot,"fewshot_cot")
# print_eval_result(label_predict_sft,"sft")




# # del pipe
# # # Reload tokenizer to save it
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=auth_token)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"



# model.push_to_hub(new_model, use_temp_dir=False)
# tokenizer.push_to_hub(new_model, use_temp_dir=False)
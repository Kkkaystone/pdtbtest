# -*- coding: utf-8 -*-
"""Fine-tune Llama 2 in Google Colab.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd

# Fine-tune Llama 2 in Google Colab
> 🗣️ Large Language Model Course

❤️ Created by [@maximelabonne](https://twitter.com/maximelabonne), based on Younes Belkada's [GitHub Gist](https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da). Special thanks to Tolga HOŞGÖR for his solution to empty the VRAM.

This notebook runs on a T4 GPU. (Last update: 24 Aug 2023)
"""
#read token
auth_token="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"

import os
import torch
# from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from implictdatareader import load_pdtb,transform_train_conversation,transform_test_conversation
from datetime import datetime
print("Fine tuning mistral 7B")
# use_flash_attention = True

# if use_flash_attention:
#     # unpatch flash attention
#     from utils.llama_patch import unplace_flash_attn_with_attn
#     unplace_flash_attn_with_attn()
# The model that you want to train from the Hugging Face hub
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# The instruction dataset to use
# dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
# new_model = "llama-2-7b-miniguanaco"
new_model = "Mistral-7B-Instruct-v0.2-7b-pdtb2.0-epoch3-p4-fix"+datetime.now().strftime("%Y%m%d%H%M%S")

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 8

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 3

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 8
#per_device_train_batch_size=6 if use_flash_attention else 4

# Batch size per GPU for evaluation
#per_device_eval_batch_size = 4
per_device_eval_batch_size = per_device_train_batch_size


# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 2

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_8bit"

# Learning rate schedule
# lr_scheduler_type = "cosine"
# Learning rate schedule (constant a bit better than cosine)

lr_scheduler_type = "constant"


# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.3

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

# Load dataset (you can process it here)
# dataset = load_dataset(dataset_name, split="train")
training_dataset=load_pdtb(split="train")
training_dataset=training_dataset.map(transform_train_conversation)
dev_dataset=load_pdtb(split="dev")
dev_dataset=dev_dataset.map(transform_train_conversation)
test_dataset=load_pdtb(split="test")
test_dataset=test_dataset.map(transform_test_conversation)

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=auth_token,
    quantization_config=bnb_config,
    device_map=device_map,
    repetition_penalty=1.5
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=auth_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
# Load LoRA configuration
# peft_config = LoraConfig(
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
#     r=lora_r,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=training_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# # Train model
trainer.train()

# # Save trained model
trainer.model.save_pretrained(new_model)

# %load_ext tensorboard
# %tensorboard --logdir results/runs

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
# prompt = "What is a large language model?"
# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

# # Empty VRAM
# del model
# # del pipe
# del trainer
# import gc
# gc.collect()
# gc.collect()

# # Reload model in FP16 and merge it with LoRA weights
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

# # Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,use_auth_token=auth_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
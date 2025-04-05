import pandas as pd
from datasets import load_dataset

import transformers

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from huggingface_hub import interpreter_login

interpreter_login()

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

train_ds = load_dataset('csv', data_files='final_df.csv', split = "train")
base_model = "microsoft/phi-2"
new_model = "phi2-ml-qa-qlora"

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = False

)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config = bnb_config,
    trust_remote_code = True,
    low_cpu_mem_usage = True,
    device_map = "auto",
)

model.config.use_cache = False
model.config.pretraining_tp = 1

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

config = LoraConfig(
  # use_dora = True,
  lora_alpha = 16,
  lora_dropout = 0.05,
  r = 32,
  bias = "none",
  task_type = "CAUSAL_LM",
  target_modules = ["Wqkv", "fc1", "fc2"]
)

model = get_peft_model(model, config)

model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir = f"./{new_model}",
    num_train_epochs = 2,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    logging_steps = 10,
    optim = "paged_adamw_8bit",
    learning_rate = 2e-4,
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.05,
    weight_decay = 0.01,
    max_steps = -1,
    load_best_model_at_end = True,
    gradient_checkpointing = True
)


val_ds = load_dataset("mjphayes/machine_learning_questions", split = "test")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts


trainer = SFTTrainer(
    model = model,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    formatting_func = formatting_prompts_func,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args = training_arguments
)

trainer.train()

model.push_to_hub(new_model)

import os
from llm import DataUtils
from peft import LoraConfig,get_peft_model
from transformers import Trainer,TrainingArguments,DataCollatorForLanguageModeling

# llm_pipeline=DataUtils.load_local_llm_pipeline(model_type=f"pretrained",model_name=f"Qwen2.5-1.5B-Instruct",task=f"text-generation")
# output=llm_pipeline(f"세종대학교에 대해 설명해줘.",max_new_tokens=100,return_full_text=False)
# print(output[0]["generated_text"])

model,tokenizer=DataUtils.load_local_llm(model_type=f"pretrained",model_name=f"Qwen2.5-1.5B-Instruct")
model.config.use_cache=False  # Trainer + gradient checkpointing 안정성

dataset=DataUtils.load_dataset(dataset_name=f"KoAlpaca-v1.1a")
dataset=dataset['train']

def preprocess_simple(example):
    prompt=example["instruction"]
    if example.get("input") and example["input"].strip()!="":
        prompt+="\n"+example["input"]
    text=prompt+"\n"+example["output"]
    return {"text":text}

dataset=dataset.map(
    preprocess_simple,
    remove_columns=dataset.column_names
)

lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

model=get_peft_model(model,lora_config)
model.print_trainable_parameters()  # trainable params 확인

# ================================
# DataCollator (Causal LM)
# ================================
data_collator=DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ================================
# TrainingArguments 설정
# ================================
OUTPUT_DIR=os.path.join('..','data','llm_basic','adapter','Adapter_LoRA_llm_Qwen2.5-1.5B-Instruct_dataset_KoAlpaca-v1.1a')

training_args=TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,                     # bf16 안 되면 fp16=True
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False
)


# ================================
# Trainer 생성 -> model이 있는 디바이스를 그대로 사용
# ================================
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ================================
# LoRA 파인튜닝 시작
# ================================
trainer.train()


# ================================
# LoRA adapter 저장 (🔥 중요)
# ================================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
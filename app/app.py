import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig,get_peft_model
from llm import DataUtils

### Tokenizer & Base Model 로드
base_model,tokenizer=DataUtils.load_pretrained_llm(model_name=f"Qwen2.5-1.5B")

### LoRA 설정 (Qwen 계열 권장)
lora_config=LoraConfig(
    r=8, # LoRA가 추가하는 저랭크 행렬의 랭크
    lora_alpha=16, # LoRA 업데이트의 스케일 계수
    lora_dropout=0.05,
    target_modules=[ # LoRA를 적용할 layer 이름
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    bias="none", # bias 파라미터를 학습할지 여부
    task_type="CAUSAL_LM", # 모델의 학습 태스크 유형
)
peft_model=get_peft_model(base_model,lora_config)
peft_model.print_trainable_parameters()

### KoAlpaca 데이터셋 로드 후 Alpaca Prompt 포맷팅
dataset=DataUtils.load_dataset(dataset_name=f"KoAlpaca-v1.1a")
train_ds=dataset["train"]

def format_alpaca(example):
    instruction=example["instruction"]
    input_text=example.get("input","") # input 없으면 공백
    output=example["output"]

    if input_text.strip(): # input 있는 경우
        text=(
            f"### Instruction:\n"
            f"{instruction}\n\n"
            f"### Input:\n"
            f"{input_text}\n\n"
            f"### Response:\n"
            f"{output}"
        )
    else:
        text=(
            f"### Instruction:\n"
            f"{instruction}\n\n"
            f"### Response:\n"
            f"{output}"
        )
    return {"text":text}

train_ds=train_ds.map(
    format_alpaca,
    remove_columns=train_ds.column_names
)

### Tokenization
def tokenize_fn(example):
    return tokenizer(
        example["text"], # Alpaca prompt 전체 문자열, Instruction + Input + Response가 모두 포함됨
        truncation=True, # 최대 길이를 초과하면 뒤를 잘라라
        max_length=1024, # 토큰 최대 길이 제한
        padding=False, # 여기서 padding no, padding은 batch 단위에서 하는 게 더 효율적 (DataCollator가 담당) 
    )

train_ds=train_ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"] # 학습에 문자열은 더이상 필요 없음
)

### Data Collater: 여러 샘플을 받아서 하나의 batch로 만들어주는 함수, Trainer는 매 스텝마다 batch=data_collator(list_of_samples) 호출
data_collator=DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # Masked Language Modeling을 사용하지 않겠다. BERT류->ㅡmim=True, GPT/Qwen/LLaMA->mim=False
)

### TrainingArguments 설정
training_args=TrainingArguments(
    output_dir="./tmp_trainer", # Trainer 내부 로그/캐시용 (adapter 저장 위치 아님)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16, # effective batch = 32
    num_train_epochs=3,
    learning_rate=2e-4, # LoRA 표준 LR
    fp16=True, # Qwen + GPU 기준
    logging_steps=50,
    save_strategy="no", # 중간 checkpoint 저장 안 함
    eval_strategy="no",
    report_to="none", # wandb 안 쓰면 none
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    remove_unused_columns=False # 중요: custom collation 시 안전
)

### Trainer 생성 및 모델 학습
trainer=Trainer(
    model=peft_model,       
    args=training_args,
    train_dataset=train_ds,
    data_collator=data_collator,
    processing_class=tokenizer,
)
trainer.train()

### save fine_tuning model
DataUtils.save_adapter(model=peft_model,tokenizer=tokenizer,adapter_name=f"LoRA_Qwen2.5-1.5B")

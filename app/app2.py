import os
from llm import DataUtils
from peft import PeftModel

model,tokenizer=DataUtils.load_local_llm(model_type=f"pretrained",model_name=f"Qwen2.5-1.5B-Instruct")

ADAPTER_PATH=os.path.join('..','data','llm_basic','adapter','Adapter_LoRA_llm_Qwen2.5-1.5B-Instruct_dataset_KoAlpaca-v1.1a')

# Load LoRA Adapter 
model=PeftModel.from_pretrained(model,ADAPTER_PATH)

# LoRA 가중치를 base model에 병합하고 adapter 제거
model=model.merge_and_unload()

MERGED_OUTPUT_PATH=os.path.join('..','data','llm_basic','fine-tunning','Qwen2.5-1.5B-Instruct_KoAlpaca-v1.1a')
model.save_pretrained(MERGED_OUTPUT_PATH)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)
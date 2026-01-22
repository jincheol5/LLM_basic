import os
import torch
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import PeftModel

class DataUtils:
    basic_path=os.path.join('..','data','llm_basic')
    @staticmethod
    def save_dataset(HF_path:str,dataset_name:str):
        """
        Input:
            HF_path
            dataset_name
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,"datasets",dataset_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

        ### load dataset from Hugging Face
        dataset=load_dataset(path=HF_path)

        ### save dataset to local dir
        dataset.save_to_disk(dataset_dict_path=dir_path)
        print(f"Save {dataset_name}!")

    @staticmethod
    def load_dataset(dataset_name:str):
        """
        Input:
            dataset_name
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,"datasets",dataset_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

        ### load dataset from local dir
        dataset=load_from_disk(dataset_path=dir_path)
        return dataset

    @staticmethod
    def save_pretrained_llm(HF_path:str,model_name:str):
        """
        Input:
            HF_path
            model_name
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,"pretrained",model_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

        ### load quantized base model from Hugging Face
        bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=HF_path,
            quantization_config=bnb_config, # for QLoRA=LoRA+quantized base model”
            dtype="auto",
            device_map="cpu", 
            low_cpu_mem_usage=True, # 메모리 사용 최소화
            trust_remote_code=True # Qwen2 커스텀 코드 실행 허용
        )

        ### load tokenizer from Hugging Face
        tokenizer=AutoTokenizer.from_pretrained(HF_path)

        ### save model and tokenizer to local dir
        model.save_pretrained(
            dir_path,
            safe_serialization=True  # safetensors 포맷으로 모델 가중치 저장 -> 순수 바이너리 포맷으로 더 빠르고 안전
        )
        tokenizer.save_pretrained(dir_path)
        print(f"Save pretrained {model_name} and its tokenizer!")
    
    @staticmethod
    def load_pretrained_llm(model_name:str):
        """
        Input:
            model_name
        Output:
            model
            tokenizer
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,"pretrained",model_name)

        ### load model from local dir
        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=dir_path,
            dtype=torch.float16,
            local_files_only=True # local에서만
        )

        ### load tokenizer from local dir 
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=dir_path,
            use_fast=True,
            local_files_only=True
        )
        return model,tokenizer

    @staticmethod
    def save_adapter(model,tokenizer,adapter_name:str):
        """
        Input:
            model (peft_model)
            tokenizer
            adapter_name
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,"adapter",adapter_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

        ### save adapter and tokenizer
        model.save_pretrained(dir_path)
        tokenizer.save_pretrained(dir_path)
        print(f"Save fine_tuning adapter and tokenizer of {adapter_name}")

    @staticmethod
    def load_adapter(base_model,adapter_name:str):
        """
        Input:
            base_model
            adapter_name
        Output:
            merged_model
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,"adapter",adapter_name)

        ### load adapter and merge with base model
        merged_model=PeftModel.from_pretrained(
            base_model,
            dir_path
        )
        return merged_model

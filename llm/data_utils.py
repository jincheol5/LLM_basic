import os
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM,AutoTokenizer

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
        dir_path=os.path.join(DataUtils.basic_path,"pretrained_llm",model_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

        ### load model from Hugging Face
        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=HF_path,
            torch_dtype="auto",
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
        dir_path=os.path.join(DataUtils.basic_path,"pretrained_llm",model_name)

        ### load model from local dir
        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=dir_path,
            torch_dtype="auto",
            device_map="cuda",
            local_files_only=True # local에서만
        )

        ### load tokenizer from local dir 
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=dir_path,
            local_files_only=True
        )
        return model,tokenizer
import os
from transformers import AutoModelForCausalLM,AutoTokenizer

class DataUtils:
    basic_path=os.path.join('..','data','llm_basic')
    @staticmethod
    def save_pretrained_llm(HF_model_name:str,model_name:str):
        """
        Input:
            HF_model_name
            model_name
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,"pretrained_llm",model_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

        ### model loading from Hugging Face
        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=HF_model_name,
            torch_dtype="auto",
            device_map="cpu", 
            low_cpu_mem_usage=True, # 메모리 사용 최소화
            trust_remote_code=True # Qwen2 커스텀 코드 실행 허용
        )

        ### tokenizer loading from Hugging Face
        tokenizer=AutoTokenizer.from_pretrained(HF_model_name)

        ### save model and tokenizer to local dir
        model.save_pretrained(
            dir_path,
            safe_serialization=True  # safetensors 포맷으로 모델 가중치 저장 -> 순수 바이너리 포맷으로 더 빠르고 안전
        )
        print(f"Save pretrained {model_name}")
        tokenizer.save_pretrained(dir_path)
        print(f"Save tokenizer for pretrained {model_name}")
    
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

        ### model loading from local dir
        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=dir_path,
            torch_dtype="auto",
            device_map="cuda",
            local_files_only=True # local에서만
        )

        ### tokenizer loading from local dir 
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=dir_path,
            local_files_only=True
        )
        return model,tokenizer
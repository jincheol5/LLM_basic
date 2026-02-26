import os
import torch
from typing_extensions import Literal
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoModel,pipeline

class DataUtils:
    basic_path=os.path.join('..','data','llm')
    
    @staticmethod
    def save_pretrained_causal_llm_from_HF(HF_path:str,model_name:str):
        """
        """
        dir_path=os.path.join(DataUtils.basic_path,"pretrained",model_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성
        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=HF_path,
            torch_dtype="auto",
            device_map="auto"
        )
        model.save_pretrained(dir_path)
        print(f"Save pretrained causal llm: {model_name}!")

    @staticmethod
    def save_pretrained_custom_llm_from_HF(HF_path:str,model_name:str):
        """
        """
        dir_path=os.path.join(DataUtils.basic_path,"pretrained",model_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성
        model=AutoModel.from_pretrained(
            HF_path, 
            trust_remote_code=True, # 다시 load 할 때도 trust_remote_code=True 필요
            torch_dtype="auto"
        )
        model.save_pretrained(dir_path)
        print(f"Save pretrained custom llm: {model_name}!")

    @staticmethod
    def save_tokenizer_from_HF(HF_path:str,model_name:str,is_custom:bool=False):
        """
        """
        dir_path=os.path.join(DataUtils.basic_path,"pretrained",model_name)
        os.makedirs(dir_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성
        if is_custom:
            tokenizer=AutoTokenizer.from_pretrained(
                HF_path,
                trust_remote_code=True
            )
        else:
            tokenizer=AutoTokenizer.from_pretrained(HF_path)
        tokenizer.save_pretrained(dir_path)
        print(f"Save tokenizer for pretrained causal llm: {model_name}!")

    @staticmethod
    def load_local_llm_pipeline(model_type:Literal['pretrained','fine_tunning']=f"pretrained",model_name:str=None,task:str=f"text-generation"):
        """
        Input:
            model_type
            model_name
        Output:
            pipeline
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,model_type,model_name)
        llm_pipeline=pipeline(
            task=task,
            model=dir_path,
            torch_dtype="auto",
            device_map="auto"
        )
        return llm_pipeline
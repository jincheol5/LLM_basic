import os
import torch
from typing_extensions import Literal
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline

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

        model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=HF_path,
            dtype="auto",
            device_map="auto"
        )

        ### load tokenizer from Hugging Face
        tokenizer=AutoTokenizer.from_pretrained(HF_path)

        ### save model and tokenizer to local dir
        model.save_pretrained(dir_path)
        tokenizer.save_pretrained(dir_path)
        print(f"Save pretrained {model_name} and its tokenizer!")
    
    @staticmethod
    def load_local_llm(model_type:Literal['pretrained','fine_tunning']="pretrained",model_name:str=None):
        """
        Input:
            model_type
            model_name
        Output:
            model
            tokenizer
        """
        ### set dir path
        dir_path=os.path.join(DataUtils.basic_path,model_type,model_name)

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
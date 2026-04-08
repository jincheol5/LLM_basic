import os
from typing_extensions import Literal
from transformers import AutoTokenizer,AutoProcessor,AutoModelForCausalLM,AutoModelForImageTextToText

class ModelUtils:
    dir_path=os.path.join("..","data","llm")

    @staticmethod
    def save_pretrained_llm_from_HF(HF_path:str,model_name:str,model_type:Literal["llm","vlm"]=f"llm",**kwargs):
        """
        **kwargs
            torch_dtype: auto or torch.bfloat16 (GPU)
            device_map: "auto", weight를 읽는 순간 바로 GPU/CPU에 분산 배치
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        os.makedirs(model_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성

        match model_type:
            case "llm":
                model=AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=HF_path,
                    torch_dtype=kwargs["torch_dtype"] ,
                    device_map=kwargs["device_map"] 
                )
            case "vlm":
                model=AutoModelForImageTextToText.from_pretrained(
                    pretrained_model_name_or_path=HF_path,
                    torch_dtype=kwargs["torch_dtype"] ,
                    device_map=kwargs["device_map"] 
                )
        model.save_pretrained(model_path)
        print(f"Save pretrained {model_type}: {model_name} from {HF_path}!")

    @staticmethod
    def save_tokenizer_from_HF(HF_path:str,model_name:str):
        """
        """
        tokenizer_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        os.makedirs(tokenizer_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=HF_path
        )
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Save tokenizer for pretrained causal llm: {model_name} from {HF_path}!")
    
    @staticmethod
    def save_processor_from_HF(HF_path:str,model_name:str):
        processor_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        os.makedirs(processor_path,exist_ok=True) # 해당 경로의 모든 폴더 없으면 생성
        tokenizer=AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=HF_path
        )
        tokenizer.save_pretrained(processor_path)
        print(f"Save processor for pretrained vlm: {model_name} from {HF_path}!")
    
    @staticmethod
    def load_local_llm(model_name:str,model_type:Literal["llm","vlm"]=f"llm"):
        """
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        match model_type:
            case "llm":
                model=AutoModelForCausalLM.from_pretrained(
                    model_path
                )
            case "vlm":
                model=AutoModelForImageTextToText.from_pretrained(
                    model_path
                )
        return model

    @staticmethod
    def load_local_tokenizer(model_name:str):
        """
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        tokenizer=AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path
        )
        return tokenizer
    
    @staticmethod
    def load_local_processor(model_name:str):
        """
        """
        model_path=os.path.join(ModelUtils.dir_path,"pretrained",model_name)
        tokenizer=AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_path
        )
        return tokenizer
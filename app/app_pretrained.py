import argparse
from modules import DataUtils

def applications(app_config:dict):
    match app_config['app_num']:
        case 1:
            """
            App 1. 
            Save pretrained model and its tokenizer from Hugging Face to local dir
            """
            DataUtils.save_pretrained_causal_llm_from_HF(HF_path=app_config['HF_path'],model_name=app_config['model_name'])
            DataUtils.save_tokenizer_from_HF(HF_path=app_config['HF_path'],model_name=app_config['model_name'])

        case 2:
            """
            App2.
            Save custom pretrained model and its tokenizer from Hugging Face to local dir
            HF_path: deepseek-ai/DeepSeek-OCR
            """
            DataUtils.save_pretrained_custom_llm_from_HF(HF_path=app_config['HF_path'],model_name=app_config['model_name'])
            DataUtils.save_tokenizer_from_HF(HF_path=app_config['HF_path'],model_name=app_config['model_name'],is_custom=True)

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--HF_path",type=str,default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model_name",type=str,default="Qwen2.5-1.5B-Instruct")
    args=parser.parse_args()

    app_config={
        # app 관련
        'app_num':args.app_num,
        'HF_path':args.HF_path,
        'model_name':args.model_name
    }
    applications(app_config=app_config)
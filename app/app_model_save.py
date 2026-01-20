import argparse
from llm import DataUtils

def applications(app_config:dict):
    match app_config['app_num']:
        case 1:
            """
            App 1. 
            Save pretrained model from Hugging Face to local dir
            """
            DataUtils.save_pretrained_llm(HF_model_name=app_config['HF_model_name'],model_name=app_config['model_name'])

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--HF_model_name",type=str,default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--model_name",type=str,default="Qwen2.5-1.5B")
    args=parser.parse_args()

    app_config={
        # app 관련
        'app_num':args.app_num,
        'HF_model_name':args.HF_model_name,
        'model_name':args.model_name
    }
    applications(app_config=app_config)
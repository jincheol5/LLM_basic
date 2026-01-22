import argparse
from llm import DataUtils

def applications(app_config:dict):
    match app_config['app_num']:
        case 1:
            """
            App 1. 
            Save dataset from Hugging Face to local dir
            """
            DataUtils.save_dataset(HF_path=app_config['HF_path'],dataset_name=app_config['dataset_name'])

if __name__=="__main__":
    """
    Execute app_train
    """
    parser=argparse.ArgumentParser()
    # app number
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--HF_path",type=str,default="beomi/KoAlpaca-v1.1a")
    parser.add_argument("--dataset_name",type=str,default="KoAlpaca_v1.1a")
    args=parser.parse_args()

    app_config={
        # app 관련
        'app_num':args.app_num,
        'HF_path':args.HF_path,
        'dataset_name':args.dataset_name
    }
    applications(app_config=app_config)
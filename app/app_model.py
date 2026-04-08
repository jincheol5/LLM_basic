import argparse
from modules import ModelUtils

def applications(**kwargs):
    match app_config['app_num']:
        case 1:
            """
            App 1. 
            Save pretrained llm and its tokenizer from Hugging Face to local dir
            """
            ModelUtils.save_pretrained_llm_from_HF(
                HF_path=kwargs["HF_path"],
                model_name=kwargs["model_name"],
                model_type=kwargs["model_type"]
            )
            match kwargs["model_type"]:
                case "llm":
                    ModelUtils.save_tokenizer_from_HF(
                        HF_path=kwargs["HF_path"],
                        model_name=kwargs["model_name"]
                    )
                case "vlm":
                    ModelUtils.save_processor_from_HF(
                        HF_path=kwargs["HF_path"],
                        model_name=kwargs["model_name"]
                    )

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--HF_path",type=str,default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model_name",type=str,default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model_type",type=str,default="llm")
    args=parser.parse_args()
    app_config={
        "app_num":args.app_num,
        "HF_path":args.HF_path,
        "model_name":args.model_name,
        "model_type":args.model_type
    }
    applications(**app_config)
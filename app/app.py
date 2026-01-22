import os
from vllm import LLM

def application():
    current_path=os.path.dirname(os.path.abspath(__file__))
    model_path=os.path.join(current_path,'..','..','data','llm_basic','pretrained_llm','Qwen2.5-1.5B')
    llm=LLM(
        model=model_path,
        attention_backend="TRITON_ATTN",  # Flash Attention 대신 Triton 사용
        enforce_eager=True
        )
    output=llm.generate(f"문학에 대해 설명해줘")
    print(output)

if __name__=="__main__":
    application()
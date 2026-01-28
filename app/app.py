from llm import DataUtils

llm_pipeline=DataUtils.load_local_llm_pipeline(model_type=f"pretrained",model_name=f"Qwen2.5-1.5B-Instruct",task=f"text-generation")
output=llm_pipeline(f"세종대학교에 대해 설명해줘.",max_new_tokens=100,return_full_text=False)
print(output[0]["generated_text"])
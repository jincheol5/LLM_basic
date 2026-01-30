from llm import DataUtils

llm_pipeline=DataUtils.load_local_llm_pipeline(model_type=f"pretrained",model_name=f"Qwen2.5-1.5B-Instruct")
while True:
    print(f"<<질문해주세요. (종료: stop 입력)>>")
    input_prompt=input(f"질문: ")
    if input_prompt==f"stop":
        break
    else:
        output=llm_pipeline(input_prompt,max_new_tokens=100,return_full_text=False)
        print(output[0]["generated_text"])
        print()
        print(f"질문: ")
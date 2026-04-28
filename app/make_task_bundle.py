import os
from mediapipe.tasks.python.genai import bundler

base_path=os.path.join("..","data")
tflite_model_path=os.path.join(base_path,"tflite","gemma-270m-it","gemma-270m-it_q8_ekv4096.tflite")
tokenizer_path=os.path.join(base_path,"llm","raw","gemma-270m-it","tokenizer.model")
task_path=os.path.join(base_path,"mediapipe","task","gemma-270m-it.task")

config=bundler.BundleConfig(
    tflite_model=tflite_model_path,
    tokenizer_model=tokenizer_path,
    start_token="<bos>",
    stop_tokens=["<eos>", "<end_of_turn>"],
    output_filename=task_path,
    
    prompt_prefix_user="<start_of_turn>user\n",
    prompt_suffix_user="<end_of_turn>\n",
    prompt_prefix_model="<start_of_turn>model\n",
    prompt_suffix_model="<end_of_turn>\n",
)
bundler.create_bundle(config)
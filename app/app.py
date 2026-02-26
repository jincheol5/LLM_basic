from modules import DataUtils

### Load model and tokenizer
model=DataUtils.load_local_llm(model_name="DeepSeek-OCR",is_custom=True)
model=model.to("cuda").eval()
tokenizer=DataUtils.load_local_tokenizer(model_name="DeepSeek-OCR",is_custom=True)

### Load image
image_file="food1.png" # 실제 이미지 경로
prompt="<image>\nFree OCR."

### inference
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
model.infer( # 리턴 값 없이 내부에서 print
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    base_size=640, 
    image_size=640, 
    crop_mode=False, 
    save_results=False,
    output_path="./results",  
    test_compress=True # 이미지를 압축 후 OCR
)
print("OCR Finished.")
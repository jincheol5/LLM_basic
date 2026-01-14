import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# 1. 설정
# =========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATASET_NAME = "kifai/KoInFoBench"   # 예시: 실제 존재하는 한국어 평가 데이터셋
SPLIT = "train"                      # 이 데이터셋은 train만 있어도 정상
NUM_SAMPLES = 5                     # 처음엔 소량으로 권장
OUTPUT_PATH = "../../data/llm_basic/base_qwen2.5_eval.json"

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# =========================
# 2. 데이터셋 로드
# =========================
dataset=load_dataset(DATASET_NAME, split=SPLIT)

print(dataset)
# 기대 출력 예:
# features: ['custom_id', 'question', 'answer']

# =========================
# 3. 모델 / 토크나이저 로드
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True
)

model.eval()

# =========================
# 4. 프롬프트 구성 함수
# =========================
def build_prompt(example):
    return f"""### Instruction:
{example['question']}

### Response:
"""

# =========================
# 5. 생성 함수
# =========================
@torch.no_grad()
def generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# =========================
# 6. 평가 실행
# =========================
results = []

for ex in dataset.select(range(NUM_SAMPLES)):
    prompt = build_prompt(ex)
    output = generate_answer(prompt)

    results.append({
        "id": ex.get("custom_id", None),
        "question": ex["question"],
        "reference_answer": ex.get("answer", None),  # 참고용 (모델 입력 ❌)
        "model_output": output
    })

# =========================
# 7. 결과 저장
# =========================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"[DONE] Saved {len(results)} samples to {OUTPUT_PATH}")

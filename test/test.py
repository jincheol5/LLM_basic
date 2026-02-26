import requests
import json

url="http://localhost:8000/v1/chat/completions"

headers={
    "Content-Type":"application/json"
}

data={
    "model":"Qwen2.5-1.5B-Instruct", # --served-model-name Qwen2.5-1.5B-Instruct
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "세종대학교에 대해 설명해줘."}
    ],
    "temperature":0.7,
    "max_tokens":1000
}

response=requests.post(url,headers=headers,data=json.dumps(data))

print("Status Code:",response.status_code)
print("Response:")
print(response.json()["choices"][0]["message"]["content"])
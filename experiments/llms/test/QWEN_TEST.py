# %% library load
import sys
import torch
import transformers
import onnx
import onnxruntime
import tokenizers
import numpy

import re
import time
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
# !pip install prompt_templates

# %% data load
with open("samples.json") as f:
    samples=json.load(f)
    
print(samples)

#%%
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",torch_dtype=torch.float16).to("cuda")



#%% prompt_templates.py
def unified_system_prompt(input_text: str) -> list:
    """
    Qwen, GPT 등 ChatML 구조 기반 LLM에 입력할 메시지 포맷 (system + user).
    분석 + 응답을 동시에 요청합니다.
    """
    system_message = {
        "role": "system",
        "content": """
당신은 사용자의 금융 발화를 분석하는 AI 송금 도우미입니다. 다음 지침에 따라 작동하세요:

1. 사용자의 문장에서 다음 항목을 추출하세요:
    - intent: 다음 중 하나 (transfer, confirm, cancel, inquiry, other, system_response)
    - amount: 숫자만 추출 (없으면 null)
    - recipient: 사람 이름 등 (없으면 null)

2. 사용자의 발화에 어울리는 자연스러운 안내 응답(response)을 생성하세요.

3. 다음 JSON 형식으로 하나의 객체로 응답하세요. 다른 텍스트는 출력하지 마세요.

예시:
{
  "intent": "transfer",
  "amount": 30000,
  "recipient": "엄마",
  "response": "엄마님께 30,000원을 송금해드릴까요?"
}
"""
    }

    user_message = {
        "role": "user",
        "content": input_text
    }

    return [system_message, user_message]


def run_inference_qwen(input_text: str, tokenizer, model, max_new_tokens=128):

    # 프롬프트 구성 (ChatML)
    messages = unified_system_prompt(input_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 토크나이즈 및 디바이스 이동
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 모델 추론
    start = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False
    )
    end = time.time()

    # 디코딩 및 프롬프트 제거
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = generated.replace(prompt, "").strip()
    
    # 5. 'assistant' 이후 텍스트만 남기기
    assistant_split = re.split(r"\bassistant\b", output_text, flags=re.IGNORECASE)
    if len(assistant_split) < 2:
        print("⚠️ 'assistant' 이후 내용을 찾지 못했습니다.")
        return output_text,None, round(end - start, 2)

    assistant_response = assistant_split[-1].strip()

    
    match = re.search(r'\{\s*"intent":.*?\}', assistant_response, re.DOTALL)
    if match:
        try:
            parsed_json = json.loads(match.group())
            return output_text,parsed_json, round(end - start, 2)
        except json.JSONDecodeError as e:
            print(f"❌ JSON 파싱 실패: {e}")
            return output_text, None, round(end - start, 2)
    else:
        print("⚠️ assistant 이후 JSON 객체를 찾을 수 없습니다.")
        return output_text, parsed_json, round(end - start, 2)

#%%
result, pasing, elapsed = run_inference_qwen("안녕", qwen_tokenizer, qwen)
print("🔍 추론 결과:", result)
print("🧩 파싱된 JSON:\n", pasing)
print("⏱️ 처리 시간:", elapsed, "초")


#%% JSON PASING & SAVE

parsed = []
raw_outputs=[]
total_time = 0

for sample in samples:
    result, parsing, elapsed = run_inference_qwen(sample["text"], qwen_tokenizer, qwen)
    total_time += elapsed

    parsed = {
        "text": sample["text"],
        "intent": parsing["intent"],
        "slots": {
            "recipient": parsing["recipient"],
            "amount": parsing["amount"]
        },
        "response": "",
        "_meta": {
            "inference_time": elapsed
        }
    }
    
    raw_outputs.append({
    "text": sample["text"],
    "raw_output": result
    })

    # if isinstance(parsed, dict):
    #     result["intent"] = parsed.get("intent")
    #     result["slots"] = parsed.get("slots", {
    #         "recipient": None,
    #         "amount": None
    #     })
    #     result["response"] = parsed.get("response", "")
    # else:
    #     result["_meta"]["error"] = "Parsing failed"

    # results.append(result)

# 저장
with open("results_qwen.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open("raw_outputs.json", "w", encoding="utf-8") as f:
    json.dump(raw_outputs, f, indent=2, ensure_ascii=False)


#%% ASSESMENT

# 평가 지표 초기화
correct_intent = 0
correct_recipient = 0
correct_amount = 0
parsing_success = 0

for result, ex in zip(results, samples):
    meta = result.get("_meta", {})
    
    # 파싱 성공 여부
    if "error" not in meta:
        parsing_success += 1

    # 예측값
    pred_intent = result.get("intent")
    pred_recipient = result.get("slots", {}).get("recipient")
    pred_amount = result.get("slots", {}).get("amount")

    # 정답값
    true_intent = ex["intent"]
    true_recipient = ex["slots"]["recipient"]
    true_amount = ex["slots"]["amount"]

    # 평가
    if pred_intent == true_intent:
        correct_intent += 1
    if pred_recipient == true_recipient:
        correct_recipient += 1
    if pred_amount == true_amount:
        correct_amount += 1

# 총 샘플 수
total = len(samples)

# 평가 결과 정리
evaluation = {
    "Intent 정확도": f"{correct_intent}/{total} ({correct_intent/total:.0%})",
    "Recipient 정확도": f"{correct_recipient}/{total} ({correct_recipient/total:.0%})",
    "Amount 정확도": f"{correct_amount}/{total} ({correct_amount/total:.0%})",
    "파싱 성공률": f"{parsing_success}/{total} ({parsing_success/total:.0%})"
}

print(evaluation)



# %%

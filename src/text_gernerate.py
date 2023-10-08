from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

model = GPT2LMHeadModel.from_pretrained(
    "skt/kogpt2-base-v2"
)
model.eval()

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    eos_token="</s>"
)

input_ids = tokenizer.encode("안녕하세요 반갑습니다", return_tensors="pt")

print(input_ids)

"""그리디 서치"""
import torch
with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        do_sample=True,          # False 이면 컨텍스트가 동일하면 결과값이 항상 같으
        min_length=0,
        max_length=300,
        no_repeat_ngram_size=3, # 토큰이 3개 이상 반복될 경우 3번째 토큰 확률을 0으로 
        #repetition_penalty=1.5,
        top_k=50,
        temperature=0.5
    )

"""토큰 익덱스를 가지고 문장생성"""

print(tokenizer.decode([el.item() for el in generated_ids[0]]))
from fetch import fetchData, getModel
from tokenizer import tokenizer

def init():
    fetchData();
    getModel();
    tokenizer();

#init();

import os 

from Korpora import Korpora
train_file = "./data/content/nsmc/train.txt"
test_file = "./data/content/nsmc/test.txt"


def getData():
    nsmc = Korpora.load("nsmc", force_download=True)

    from config import args
    def write_lines(path, lines):
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(f'{line}\n')
    """순수 텍스트 형태로 저장"""
    write_lines(train_file ,nsmc.train.get_all_texts())
    write_lines(test_file, nsmc.test.get_all_texts())


"""바이트수준 BPE 어휘집합 구축"""
from tokenizers import ByteLevelBPETokenizer
def getByteLevelBPEModel():
    bytebpe_tokenizer = ByteLevelBPETokenizer();
    bytebpe_tokenizer.train(
        files = [train_file, test_file],
        vocab_size= 10000, # 어휘크기
        special_tokens=["[PAD]"] # 특수 토큰 추가
    )
    bytebpe_tokenizer.save_model("./data/model/nsmc/bbpe")



"""워드피스 어휘집합 구축"""
from tokenizers import BertWordPieceTokenizer

def getWordpieceModel():
    wordpiece_tokenizer = BertWordPieceTokenizer(lowercase=False)
    wordpiece_tokenizer.train(
        files=[train_file, test_file],
        vocab_size= 10000, # 어휘크기
    )
    wordpiece_tokenizer.save_model("./data/model/nsmc/wordpiece")

#getByteLevelBPEModel()
#getWordpieceModel()

"""GPT 토크나이저로 토큰화하기"""
from transformers import GPT2Tokenizer
tokenizer_gpt = GPT2Tokenizer.from_pretrained("./data/model/nsmc/bbpe")
tokenizer_gpt.pad_token = "[PAD]"
sentences = [
    "아 더빙. 진짜 짜증나네요 목소리",
    "흠 포스터보고 초팅영화인줄... 오버연기조차 가볍지 않네",
    "별루 였다..."
]
tokenizer_sentences = [tokenizer_gpt.tokenize(sentence) for sentence in sentences]

batch_inputs = tokenizer_gpt(
    sentences,
    padding="max_length", # 문장최대 길이에 맞춰 패딩
    max_length= 12, # 문장의 토큰 기준 최대 길이
    truncation=True
)
batch_inputs
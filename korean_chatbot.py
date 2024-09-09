import model
import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import re
import urllib.request
# urllib.request.urlretrieve(
#     "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", 
#     filename="ChatBotData.csv")

train_data = pd.read_csv('ChatBotData.csv')
print(train_data.head())
tokenizer = get_tokenizer('spacy', language='ko_core_news_sm')

## https://tutorials.pytorch.kr/beginner/torchtext_translation.html
def build_vocab(strings, tokenizer):
    counter = Counter()
    for string_ in strings:
        counter.update(tokenizer(string_))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def data_process(questions: list[str], answers: list[str]):
    data = []
    for (raw_q, raw_ans) in zip(questions, answers):
        q_tensor_ = torch.tensor([vocab[token] for token in tokenizer(raw_q)],
                                dtype=torch.long)
        ans_tensor_ = torch.tensor([vocab[token] for token in tokenizer(raw_ans)],
                                dtype=torch.long)
        data.append((q_tensor_, ans_tensor_))
    return data

questions = []
answers = []
for (sentenceQ, sentenceA) in zip(train_data['Q'], train_data['A']):
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentenceQ = re.sub(r"([?.!,])", r" \1 ", sentenceQ)
    sentenceQ = sentenceQ.strip()
     
    sentenceA = re.sub(r"([?.!,])", r" \1 ", sentenceA)
    sentenceA = sentenceA.strip()
    
    questions.append(sentenceQ)
    answers.append(sentenceA)
    

vocab = build_vocab(questions + answers, tokenizer)
PAD_IDX = vocab['<pad>']
BOS_IDX = vocab['<bos>']
EOS_IDX = vocab['<eos>']

train_data = data_process(questions, answers)

print(train_data[0])


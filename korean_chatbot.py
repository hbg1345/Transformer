import model
import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer
import torch.nn.functional as F
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
        # add start and end token into both sides of sequence
        q_tensor_ = torch.tensor([BOS_IDX] + [vocab[token] for token in tokenizer(raw_q)] + [EOS_IDX],
                                dtype=torch.long)
        ans_tensor_ = torch.tensor([BOS_IDX] + [vocab[token] for token in tokenizer(raw_ans)] + [EOS_IDX],
                                dtype=torch.long)
        # padding tensor for fixed sequence length
        q_tensor_ = F.pad(q_tensor_, (0, model.seq_len - q_tensor_.shape[0]), 'constant', PAD_IDX)
        ans_tensor_ = F.pad(ans_tensor_, (0, model.seq_len - ans_tensor_.shape[0]), 'constant', PAD_IDX)
        
        data.append((q_tensor_, ans_tensor_))
        
    return data


questions = []
answers = []
# https://wikidocs.net/89786
for (sentenceQ, sentenceA) in zip(train_data['Q'], train_data['A']):
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentenceQ = re.sub(r"([?.!,])", r" \1 ", sentenceQ)
    sentenceQ = sentenceQ.strip()
     
    sentenceA = re.sub(r"([?.!,])", r" \1 ", sentenceA)
    sentenceA = sentenceA.strip()
    
    questions.append(sentenceQ)
    answers.append(sentenceA)

    
def padding_mask(x):
    """Create padding mask
    x: sequence of word (shape: seq_len)
    Returns:
    mask: mask where i-th column is 1 if x[i]==PAD_IDX, 0 otherwise (shape: seq_len x seq_len)
    """
    mask = x == PAD_IDX
    mask = mask.long()
    mask = mask.expand((mask.shape[0], mask.shape[0]))
    return mask

def look_ahead_mask(x):
    """Create look ahead mask
    x: sequence of word (shape: seq_len)
    Returns:
    mask: mask[i][j] = 1 if j > i, 0 otherwise (shape: seq_len x seq_len)
    example:
    x = [1 2 3 4] 
    mask = 
    [0 1 1 1]
    [0 0 1 1]
    [0 0 0 1]
    [0 0 0 0]
    """
    return torch.triu(torch.ones((x.shape[0], x.shape[0])), 1)

vocab = build_vocab(questions + answers, tokenizer)
PAD_IDX = vocab['<pad>']
BOS_IDX = vocab['<bos>']
EOS_IDX = vocab['<eos>']

train_data = data_process(questions, answers)

x = train_data[0][0]
print(padding_mask(x))
print(look_ahead_mask(x))

# <TODO> 
# process data to enable traing
# </TODO>

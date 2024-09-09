import random
import torch
import torch.nn as nn

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_device(device)
seed = 2024 ## set random seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

## below line may slow down training but ensure deterministic result.
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

d_model = 512 # dimension of an embedding vector.
N = 6 # num of encoder, decoder sublayers.
C = 10000 # constant in positinal encoding.
h = 8 # num of attetnion heads.
vocab_size = 37000 # size of vocabulary(source == target)
seq_len = 40 ## number of words in a sequence
INF = 1e10
dff = 2048

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model, C):
        super(EmbeddingLayer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.constant = torch.tensor(C)
        self.embedd = nn.Embedding(vocab_size, d_model)
        # self.embedd.requires_grad_(False)
        
        ## positional encoding matrix
        self.pos_mat = torch.zeros(self.seq_len, self.d_model) 
        for pos in range(self.seq_len):
            for j in range(self.d_model):
                f = torch.cos
                if (j+1) % 2: f = torch.sin
                self.pos_mat[pos][j] = f((pos+1) / torch.pow(self.constant, (j+1)//2/self.d_model))
                
    def forward(self, x):
        # x.shape: (batch, seq_len)
        x = self.embedd(x) # (batch, seq_len, d_moel)
        x += self.pos_mat # broadcasting
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, mask):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = torch.tensor(d_model // n_heads)
        self.h = n_heads
        self.WQ = []
        self.WK = []
        self.WV = []
        # d_model==h*dv -> d_model
        self.WO = nn.Linear(d_model, d_model)
        # padding and look ahead mask(only for decoder)
        self.mask = mask # (d_model, d_model)
        
        for _ in range(self.h):
            self.WQ.append(nn.Linear(self.d_model, self.d_k))
        for _ in range(self.h):
            self.WK.append(nn.Linear(self.d_model, self.d_k))
        for _ in range(self.h):
            self.WV.append(nn.Linear(self.d_model, self.d_k))
    
    def attention(self, Q, K, V):
        # (batch, seq_len, d_k)
        score = torch.matmul(Q, torch.transpose(K, 1, 2)) # Q * K^T
        # print(score.shape)
        score /= torch.sqrt(self.d_k) # scaling
        score += self.mask
        score = nn.functional.softmax(score, dim=2)
        return torch.matmul(score, V) # (batch, seq_len, d_k)
        
    def forward(self, Q, K, V):
        ## Query, Key, Value matrix
        ## (bacth, seq_len, d_model)
        out = torch.empty(0)
        
        for i in range(self.h):
            nQ = self.WQ[i](Q)
            nK = self.WK[i](K)
            nV = self.WV[i](V)
            
            out = torch.concat((out, self.attention(nQ, nK, nV)), dim=2)
            
        ## (bacth, seq_len, h*d_k)
        out = self.WO(out)
        ## (bacth, seq_len, d_model)
        return out 

class FFN(nn.Module):
    def __init__(self, d_ff, d_model):
        super(FFN, self).__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = nn.functional.relu(self.w1(x))
        x = self.w2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, mask):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, mask)
        self.norm1 = nn.LayerNorm((seq_len, d_model))
        self.norm2 = nn.LayerNorm((seq_len, d_model))
        self.ffn = FFN(d_ff, d_model)
        
    def forward(self, x):
        x += self.attention(x, x, x) # self attention
        x = self.norm1(x)
        x += self.ffn(x)
        x = self.norm2(x)
        return x

class DeepEncoderLayer(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, n_heads, mask):
        super(DeepEncoderLayer, self).__init__()
        self.enc = []
        for _ in range(num_layers):
            self.enc.append(EncoderLayer(d_model, d_ff, n_heads, mask))
        self.N = num_layers
    
    def forward(self, x):
        for i in range(self.N):
            x = self.enc[i](x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, mask, enc_out, enc_mask):
        super(DecoderLayer, self).__init__()
        self.enc_out = enc_out
        ## Decoder self attention
        self.masked_attention = MultiHeadAttention(d_model, n_heads, mask)
        ## (Key, Value) from encoder, Query from decoder.
        self.cross_attention = MultiHeadAttention(d_model, n_heads, enc_mask)
        self.norm1 = nn.LayerNorm((seq_len, d_model))
        self.norm2 = nn.LayerNorm((seq_len, d_model))
        self.norm3 = nn.LayerNorm((seq_len, d_model))        
        self.ffn = FFN(d_ff, d_model)
        
    def forward(self, x):
        x += self.masked_attention(x, x, x) # self attention
        x = self.norm1(x)
        x += self.cross_attention(x, self.enc_out, self.enc_out)
        x = self.norm2(x)
        x += self.ffn(x)
        x = self.norm3(x)
        return x
    
class DeepDecoderLayer(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, n_heads, mask, enc_out, enc_mask):
        super(DeepDecoderLayer, self).__init__()
        self.dec = []
        for _ in range(num_layers):
            self.dec.append(DecoderLayer(d_model, d_ff, n_heads, mask, enc_out, enc_mask))
        self.N = num_layers
    
    def forward(self, x):
        for i in range(self.N):
            x = self.dec[i](x)
        return x


## test
# d_model = 4
# d_ff = 8
# n_heads = 2
# vocab_size = 3
# seq_len = 10
# batch = 16
# N = 2
# mask = torch.randint(0, 2, (seq_len, seq_len)) 
# a = torch.randint(0, vocab_size, (batch, seq_len))
# print(a.shape, "batch, seq_len")
# ## embedding
# embed = EmbeddingLayer(vocab_size, seq_len, d_model, C)
# a = embed(a)
# print(a.shape, "batch, seq_len, d_model")
# ## encoder layer
# enc = DeepEncoderLayer(N, d_model, d_ff, n_heads, mask)
# a = enc(a)
# print(a.shape, "batch, seq_len, d_model")
# dec = DeepDecoderLayer(N, d_model, d_ff, n_heads,mask,a,mask)
# a = dec(a)

# print(a.shape, "batch, seq_len, d_model")






















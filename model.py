import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import numpy as np
import torch
#import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import torch



class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, decoder, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        #self.encoder = encoder
        self.encoder = VisionTransformer(num_layers=12)
        self.decoder = decoder
        #self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.output= nn.Linear(512, 2)
    def forward(self, src, tgt, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.output(self.decode(self.encode(src), tgt, tgt_mask))


    def encode(self, src):
        return self.encoder(src)

    def decode(self, memory, tgt, tgt_mask): # memory (b, 256, 768)
        return self.decoder(self.tgt_embed(tgt), memory, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)  ######## 这需要改####################################################################


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# class Encoder(nn.Module):
#     "Core encoder is a stack of N Encoderlayers"
#
#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = nn.LayerNorm(layer.size)
#
#     def forward(self, x, mask):
#         "Pass the input (and mask) through each layer in turn."
#         for layer in self.layers:
#             x = layer(x, mask)
#         return self.norm(x)
class SublayerConnection(nn.Module):
    """
    Apply for EncoderLayer/DecoderLayer
    sublayer: multi-head attention/dense structure
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
# class EncoderLayer(nn.Module):
#     # EncoderLayer 2 layer
#     "EncoderLayer is made up of self-attn and feed forward (defined below)"
#     def __init__(self, size, self_attn, feed_forward, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 2)
#         self.size = size
#
#     def forward(self, x, mask):
#         "Follow Figure 1 (left) for connections."
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.feed_enc_dec = nn.Linear(768, 512) ##################################################
    def forward(self, x, memory, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, self.feed_enc_dec(m), self.feed_enc_dec(m)))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    #print(subsequent_mask)
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        print(d_model, h)
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) ## Linear : (512, 512)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # The output q, k, v dim is (batch_size, N, D=512) --> (batch_size, head, N, 64)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        #return self.lut(x) * math.sqrt(self.d_model)

        return self.lut(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

########################################  The main model of trajectory prediction Transformer model  ##########################################################
def make_model(tgt_vocab=2, N=3, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # 随机初始化参数，这非常重要
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)






class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches , emb_dim))  #(1, 257, 768)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding  # broadcast: EMbedding- (b, N+1, D) + PE- (1, N+1, D) (b, 257, 768)

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim)) # (768, 12, 64)
        self.bias = nn.Parameter(torch.zeros(*feat_dim)) # (12, 64)

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias  # 任意两个维度的矩阵，按着给定的维度做矩阵相乘
        return a


class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads  # 12   head数量
        self.head_dim = in_dim // heads # 64 每一个head维度
        self.scale = self.head_dim ** 0.5 # 8

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim)) # (768, 12, 64)
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))  # (768, 12, 64)
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim)) #
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,)) # (12, 64, 768)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))  # x: (b, 257, 768) * w: (768, 12, 64) --> q: (b, 257, 12, 64)
        k = self.key(x, dims=([2], [0]))    # x: (b, 257, 768) * w: (768, 12, 64) --> q: (b, 257, 12, 64)
        v = self.value(x, dims=([2], [0]))  # x: (b, 257, 768) * w: (768, 12, 64) --> q: (b, 257, 12, 64)

        q = q.permute(0, 2, 1, 3)  # (b, 12, 257, 64)  # k.transpose(-2, -1): (b, 12, 64, 257)
        k = k.permute(0, 2, 1, 3)  # (b, 12, 257, 64)
        v = v.permute(0, 2, 1, 3)  # (b, 12, 257, 64)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # 矩阵相乘broadcast (b, 12, 257, 257)
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v) #  (b, 12, 257, 257) * (b, 12, 257, 64) --> (b, 12, 257, 64)
        out = out.permute(0, 2, 1, 3) # (b, 257, 12, 64)

        out = self.out(out, dims=([2, 3], [0, 1])) # (b, 257, 768)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate) #  加入位置编码后的embedding (b, 256, 768)

        # encoder blocks
        in_dim = emb_dim # 768
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):

        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out)

        out = self.norm(out)
        return out


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 image_size=(512, 512),
                 patch_size=(32, 32),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=0,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # embedding layer
        fh, fw = patch_size # (16, 16)
        gh, gw = h // fh, w // fw # (16, 16)
        num_patches = gh * gw # 256
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw)) # (b, 768, 16, 16)
        # class token
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim)) # cls: (1,1,768)

        # transformer
        self.transformer = Encoder(
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        # classfier
        # self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x): # x shape (b, 3, 256, 256)
        emb = self.embedding(x)     # (n, c, gh, gw)  (b, 768, 16, 16)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c) (b, 16, 16, 768)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)  # (batchsize, N: patch_number, D: embedding_dim) # (b, 256, 768)

        # prepend class token
        # cls_token = self.cls_token.repeat(b, 1, 1) # class token 扩展其batch_size维度 # (b, 1, 768)
        # emb = torch.cat([cls_token, emb], dim=1)  # (batchsize, N+1: patch_number, D: embedding_dim) # (b, 257, 768)

        # transformer
        feat = self.transformer(emb)

        # classifier
        #logits = self.classifier(feat[:, 0])
        memory = feat   # (b, 256, 768)

        return memory


if __name__ == '__main__':
    #model = VisionTransformer(num_layers=2)
    # out = model(x)

    # x = torch.randn((4, 3, 512, 512)).float().cuda()
    # y = torch.randn((4, 249, 2)).float().cuda()
    #
    # #print(x)
    # tgt_mask = subsequent_mask(249)
    # tgt_mask = tgt_mask.float().cuda()
    # model_final = make_model(2, 12)
    # model_final.to('cuda')
    # output = model_final.forward(x, y, tgt_mask)
    # print(output.size())




    model2 = make_model(10,0)
    state_dict = model2.state_dict()


    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))

    #print(model2.parameters())









import copy  # 导入深拷贝模块，用于复制层结构
import torch  # 导入 PyTorch 核心库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式接口（如 softmax, relu, log_softmax）
import math  # 导入数学函数库，用于 sqrt, log, exp 等

# 判断是否可用 GPU，否则使用 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 辅助函数：深度克隆模块 N 次，返回 ModuleList
def clones(module, N):
    """产生 N 个相同的模块（深度拷贝）"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ----------------------------------------------------------------------
# 词嵌入层：将词汇索引转为 d_model 维向量，并乘以 sqrt(d_model)
class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        # 标准嵌入层，输入 vocab 大小，输出 d_model 维
        self.lut = nn.Embedding(vocab, d_model)
        # 保存模型维度，用于缩放
        self.d_model = d_model

    def forward(self, x):
        # 乘以 sqrt(d_model) 是为了缩放嵌入，使后续位置编码的加和方差稳定
        return self.lut(x) * math.sqrt(self.d_model)


# ----------------------------------------------------------------------
# 位置编码层：为正弦/余弦函数生成的位置信息，加到词嵌入上
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        # 丢弃层，按一定概率随机置零，防止过拟合
        self.dropout = nn.Dropout(p=dropout)

        # 预先生成一个位置编码矩阵 pe，形状 (max_len, d_model)
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 位置索引，形状 (max_len, 1)
        position = torch.arange(0, max_len, device=DEVICE).unsqueeze(1)
        # 计算频率项的分母部分：10000^(2i/d_model), 形状 (d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model)
        )
        # 偶数位置用 sin，奇数位置用 cos(广播机制),形状 (max_len, d_model/2)，pe形状 (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加 batch 维度，形状 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 当模型被 model.to(DEVICE) 时，self.pe 会自动移到 GPU 上，无需手动处理。
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x 形状: (batch, seq_len, d_model)
        # 将对应长度的位置编码加到输入上（自动广播）
        x = x + self.pe[:, : x.size(1)]
        # 应用 dropout 并返回
        return self.dropout(x)


# ----------------------------------------------------------------------
# 缩放点积注意力核心函数
def attention(query, key, value, mask=None, dropout=None):
    # query/key/value 形状: (batch, n_head, seq_len, d_k)
    d_k = query.size(-1)  # 每个头的维度

    # 计算 Q·K^T，并缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果有掩码（如 padding 或未来位置），将对应位置设为很小的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 对最后一维（词维度）做 softmax，得到注意力权重
    p_attn = F.softmax(scores, dim=-1)
    # 可选 dropout 随机丢弃部分注意力权重
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 权重与 V 相乘，返回加权结果以及注意力权重（用于可视化）
    return torch.matmul(p_attn, value), p_attn


# ----------------------------------------------------------------------
# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        # 确保 d_model 可以被 n_head 整除
        assert d_model % n_head == 0
        self.d_k = d_model // n_head  # 每个头的维度
        self.n_head = n_head
        # 4 个线性变换：分别用于 Q, K, V 以及最后的输出投影
        self.liners = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # 保存注意力权重，用于分析
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 如果提供了 mask，则扩展一个维度用于多头（每个头共享相同 mask）
        if mask is not None:
            mask = mask.unsqueeze(1)  # shape: (batch, 1, seq_len, seq_len)

        nbatches = query.size(0)  # batch size

        # 1) 对 Q, K, V 分别做线性变换，然后切分成 n_head 个头，并转置
        #    - 先线性: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        #    - view 拆分: (batch, seq_len, n_head, d_k)
        #    - transpose: (batch, n_head, seq_len, d_k)
        query, key, value = [
            l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for l, x in zip(self.liners, (query, key, value))
        ]

        # 2) 应用注意力函数
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 合并多头：转置回 (batch, seq_len, n_head, d_k)，然后连续 view 为 (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)

        # 4) 最后一个线性变换（输出投影）
        return self.liners[-1](x)


# ----------------------------------------------------------------------
# 层归一化（与 nn.LayerNorm 类似，但显式实现以便理解）
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        # 可学习的缩放因子和偏移量
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps  # 防止除零的小量

    def forward(self, x):
        # 计算最后一维的均值和标准差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 归一化并线性变换
        return self.gamma * (x - mean) / torch.sqrt(std**2 + self.eps) + self.beta


# ----------------------------------------------------------------------
# 位置前馈网络 (FFN)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 第一个线性层将维度从 d_model 扩展到 d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 第二个线性层压缩回 d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # FFN(x) = W_2 * dropout(ReLU(W_1 * x))
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ----------------------------------------------------------------------
# 子层连接结构：残差连接 + 层归一化 + 子层本身
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 先归一化，再应用于子层，然后 dropout，最后残差相加
        return x + self.dropout(sublayer(self.norm(x)))


# ----------------------------------------------------------------------
# 编码器层：包含自注意力和前馈网络，每个子层后都有残差连接
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 两个子层连接：一个用于自注意力，一个用于前馈网络
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 第一个子层：多头自注意力（Q=K=V=x）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个子层：前馈网络
        return self.sublayer[1](x, self.feed_forward)


# ----------------------------------------------------------------------
# 编码器：由 N 个编码器层堆叠而成
class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        # 复制 N 个相同的层
        self.layers = clones(layer, n_layers)
        # 最后的层归一化
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # 逐层传递
        for layer in self.layers:
            x = layer(x, mask)
        # 最终归一化后返回
        return self.norm(x)


# ----------------------------------------------------------------------
# 解码器层：包含自注意力、编码器-解码器注意力、前馈网络
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_atten, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn  # 目标序列自注意力（带掩码防止看到未来）
        self.src_atten = src_atten  # 编码器-解码器注意力（Q 来自目标，K,V 来自编码器）
        self.feed_forward = feed_forward
        # 三个子层连接
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 第一子层：带掩码的自注意力（只允许看到已生成的词）
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二子层：编码器-解码器注意力，K,V 来自编码器输出 memory，Q 来自当前目标序列
        x = self.sublayer[1](x, lambda x: self.src_atten(x, memory, memory, src_mask))
        # 第三子层：前馈网络
        return self.sublayer[2](x, self.feed_forward)


# ----------------------------------------------------------------------
# 解码器：由 N 个解码器层堆叠而成
class Decoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 逐层处理
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # 最终归一化
        return self.norm(x)


# ----------------------------------------------------------------------
# 生成器：将解码器输出映射为词表大小的概率分布
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 应用 log_softmax 得到对数概率（用于交叉熵损失）
        return F.log_softmax(self.proj(x), dim=-1)


# ----------------------------------------------------------------------
# 完整的 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # 源语言词嵌入 + 位置编码
        self.tgt_embed = tgt_embed  # 目标语言词嵌入 + 位置编码
        self.generator = generator

    def encode(self, src, src_mask):
        # 对源序列进行词嵌入+位置编码，然后通过编码器
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # 对目标序列进行嵌入，结合编码器输出进行解码
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 标准前向传播：编码 -> 解码
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


# ----------------------------------------------------------------------
# 模型构建函数：设置超参数并返回初始化的模型
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1
):
    c = copy.deepcopy  # 别名，用于深拷贝
    # 创建多头注意力和前馈网络实例（会被多次深拷贝）
    attn = MultiHeadAttention(n_heads, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # 组装 Transformer
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # 参数初始化：对维度大于 1 的参数使用 Xavier 均匀初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 将模型移动到指定设备（GPU/CPU）
    return model.to(DEVICE)

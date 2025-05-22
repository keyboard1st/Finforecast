import torch.nn as nn
import torch
from utils.log import Config
import os
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            input_dim: 输入特征维度 D
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 层数
            dropout: 层间Dropout概率
        """
        super().__init__()
        self.embedding = nn.Linear(config.input_dim, config.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead= 2,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers)
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)
        self.pos_encoder = PositionalEncoding(config.hidden_dim)

    def generate_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

    def forward(self, x):
        emb = self.embedding(x)
        seq_len = emb.size(1)
        mask = self.generate_mask(seq_len)
        # x.shape:[B,T,C]
        output = self.encoder(emb, mask=mask)

        # 全连接层预测收益率
        pred = self.fc(output)  # 形状 [batch_size, output_dim]
        return pred.squeeze()


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)

        self.register_buffer("sin", sinusoid_inp.sin().view(max_seq_len, dim // 2, 1))
        self.register_buffer("cos", sinusoid_inp.cos().view(max_seq_len, dim // 2, 1))

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        """ x: [B, T, H] """
        seq_len = x.size(1)
        sin = self.sin[:seq_len].view(1, seq_len, self.dim // 2, 1)
        cos = self.cos[:seq_len].view(1, seq_len, self.dim // 2, 1)

        x_rot = x.view(*x.size()[:-1], self.dim // 2, 2)
        x_rot = (x_rot * cos) + (self._rotate_half(x_rot) * sin)
        return x_rot.view(*x.size())


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # RoPE模块
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, H]
            mask: [T, T]
        Returns:
            [B, T, H]
        """
        B, T, H = x.size()

        # 线性投影
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, T, hd]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, T, hd]
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, nh, T, hd]

        # 应用RoPE
        q = self.rope(q)  # [B, nh, T, hd]
        k = self.rope(k)  # [B, nh, T, hd]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, nh, T, T]
        if mask is not None:
            attn += mask[None, None, :, :]  # 广播掩码

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, T, H)  # [B, T, H]
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(hidden_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力分支
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # 前馈分支
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class FinancialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.proj = nn.Linear(config.hidden_dim, 1)  # 输出单个预测值


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def generate_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            [B]  # 每个样本预测一个值
        """
        # 维度验证
        B, T, C = x.size()

        # 嵌入层 [B,T,C] => [B,T,H]
        x = self.embed(x)  # H=hidden_dim

        # 生成因果掩码
        mask = self.generate_mask(T).to(x.device)

        # 通过所有Transformer层
        for block in self.blocks:
            x = block(x, mask)

        # 关键修改：取最后一个时间步的特征 [B,T,H] => [B,H]
        last_hidden = x[:, -1, :]  # 只取序列末端

        # 输出投影 [B,H] => [B,1] => [B]
        return self.proj(last_hidden).squeeze(-1)


if __name__=='__main__':

    torch.manual_seed(42)

    # 测试参数
    batch_size = 3280
    seq_len = 30
    input_dim = 68
    hidden_dim = 64
    output_dim = 1
    num_layers = 3
    dropout = 0.0
    class config:
        input_dim = input_dim
        hidden_dim = hidden_dim
        num_layers = num_layers
        output_dim = output_dim
        dropout = dropout


    # 初始化模型
    model = Encoder(config)

    # 创建随机输入数据
    x = torch.randn(batch_size, seq_len, input_dim)  # 形状 [batch_size, seq_len, input_dim]

    # 切换模型到评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        output = model(x)

    # 打印输出形状和部分输出值
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例 (前5个): {output[:5]}")

    # 验证输出形状是否符合预期
    expected_shape = (batch_size,) if output_dim == 1 else (batch_size, output_dim)
    assert output.shape == expected_shape, f"输出形状不正确，应为 {expected_shape}，但实际为 {output.shape}"
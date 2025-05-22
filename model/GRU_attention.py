import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

class FeatureSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super().__init__()
        self.feature_size = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert self.head_dim * num_heads == input_dim, "feature_size必须能被num_heads整除"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.out = nn.Linear(input_dim, input_dim)

        self.scale_factor = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # 生成Q, K, V (B, T, C)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 多头处理
        Q = Q.reshape(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        K = K.reshape(B, T, H, D).permute(0, 2, 3, 1)  # (B, H, D, T)
        V = V.reshape(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K) * self.scale_factor  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)  # (H, B*T, D)
        out = out.permute(1, 0, 2).contiguous().reshape(B * T, C)  # (B*T, C)
        out = self.out(out).reshape(B, T, C)
        # 恢复原始维度

        return out


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.W_Q = nn.Linear(input_dim, input_dim)
        self.W_K = nn.Linear(input_dim, input_dim)
        self.W_V = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        Q = self.W_Q(x)  # (B, T, C)
        K = self.W_K(x)  # (B, T, C)
        V = self.W_V(x)  # (B, T, C)

        # Step 2: 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, T, T)
        scores = scores / torch.sqrt(torch.tensor(self.input_dim))  # 缩放

        # Step 3: Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # Step 4: 加权求和
        output = torch.matmul(attn_weights, V)  # (B, T, C)
        return output


class AttGRU(nn.Module):
    def __init__(self, config):
        """
        Args:
            input_dim: 输入特征维度 D
            hidden_dim: GRU隐藏层维度
            output_dim: 输出维度
            num_layers: GRU层数
            dropout: 层间Dropout概率
        """
        super().__init__()
        self.att = SelfAttention(input_dim=config.window_size)
        self.fc1 = nn.Linear(config.window_size,config.window_size)
        # self.position_emb = nn.Parameter(
        #     torch.zeros(1, config.input_dim, config.window_size)  # [1, C, T]
        # )
        # nn.init.xavier_normal_(self.position_emb)
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=False,  # 输入形状为 [seq_len, batch, input_size]
            dropout=config.dropout
        )
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(config.hidden_dim // 2, config.output_dim)
        self.model_name = 'AttGRU'

    def forward(self, x):
        x_t = x.permute(0, 2, 1) # [B, T, C] -> [B, C, T]
        # x_t = x_t + self.position_emb
        # print(self.position_emb.flatten().data.mean())
        # print(self.position_emb.flatten().data.std())
        # top_values, top_indices = torch.topk(self.position_emb.flatten(), k=10)
        # print("Top 10 position indices:", top_indices.cpu().numpy())
        # x_debug = x_t[0, :10, 0].detach().cpu().numpy()
        # print("ori x:", x_debug)
        # print("Corresponding values:", top_values.detach().cpu().numpy())
        att_x = self.att(x_t) # [B, C, T]
        gate_weights = F.softmax(self.fc1(x_t), dim=-1)   # [B, C, T] → softmax在dim=-1（T）
        x_t = x_t + gate_weights * att_x  # [B, C, T]
        x = x_t.permute(2, 0, 1)
        # 转换为GRU需要的形状: [seq_len, batch_size, input_dim]
        # GRU输出:
        # - output: 所有时间步的隐藏状态 [seq_len, batch, hidden_dim]
        # - h_n: 最后一个时间步的隐藏状态 [num_layers, batch, hidden_dim]
        output, h_n = self.gru(x)
        # last_hidden =  torch.cat((h_n[-2, :, :], h_n[-1, : , : ]), dim=-1)
        # 取最后一层的最终隐藏状态（若多层GRU）
        last_hidden = h_n[-1, :, :]  # 形状 [batch_size, hidden_dim]
        # 全连接层预测收益率
        pred = self.relu(self.fc2(last_hidden))
        pred = self.fc3(pred)
        return pred.squeeze()

# 使用示例
if __name__ == "__main__":
    B, T, C = 3873, 30, 312  # 批次大小, 时间步长, 特征维度
    x = torch.randn(B, T, C)

    class TestConfig:
        def __init__(self):
            self.input_dim = 312  # 输入特征维度
            self.window_size = 30
            self.hidden_dim = 128  # GRU隐藏层维度
            self.output_dim = 1  # 输出维度
            self.num_heads = 4  # 自注意力头数
            self.num_layers = 3  # GRU层数
            self.dropout = 0.2  # Dropout概率



    config = TestConfig()
    model = AttGRU(config)
    output = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)

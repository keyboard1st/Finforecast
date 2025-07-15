import torch
import torch.nn as nn
import math
import time


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    基于论文: RoFormer: Enhanced Transformer with Rotary Position Embedding
    """
    
    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        
        # 计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            rotated x: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # 生成位置索引
        position = torch.arange(seq_len, device=x.device).float()
        
        # 计算频率矩阵 [seq_len, d_model//2]
        freqs = torch.outer(position, self.inv_freq)
        
        # 生成cos和sin [seq_len, d_model//2]
        cos_freqs = freqs.cos()
        sin_freqs = freqs.sin()
        
        # 扩展维度以匹配batch维度 [1, seq_len, d_model//2]
        cos_freqs = cos_freqs.unsqueeze(0)
        sin_freqs = sin_freqs.unsqueeze(0)
        
        # 分离奇偶维度
        x_even = x[..., 0::2]  # [batch_size, seq_len, d_model//2]
        x_odd = x[..., 1::2]   # [batch_size, seq_len, d_model//2]
        
        # 应用旋转变换
        rotated_even = x_even * cos_freqs - x_odd * sin_freqs
        rotated_odd = x_even * sin_freqs + x_odd * cos_freqs
        
        # 重新组合 [batch_size, seq_len, d_model]
        rotated_x = torch.zeros_like(x)
        rotated_x[..., 0::2] = rotated_even
        rotated_x[..., 1::2] = rotated_odd
        
        return rotated_x


class RoPETransformerEncoderLayer(nn.Module):
    """
    集成RoPE的Transformer Encoder层
    自动确保因果性（不使用未来信息）
    """
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除"
        
        # Query, Key, Value 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RoPE位置编码
        self.rope = RoPEPositionalEncoding(self.head_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def _generate_causal_mask(self, seq_len, device):
        """生成因果掩码确保不使用未来信息"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.size()
        
        # 1. 自注意力分支
        residual = x
        x = self.norm1(x)
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # 应用RoPE到q和k
        q_rope = torch.zeros_like(q)
        k_rope = torch.zeros_like(k)
        
        for head in range(self.nhead):
            q_rope[:, :, head, :] = self.rope(q[:, :, head, :])
            k_rope[:, :, head, :] = self.rope(k[:, :, head, :])
        
        # 重塑为 [batch_size, nhead, seq_len, head_dim]
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑回 [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_output = self.out_proj(attn_output)
        
        # 残差连接
        x = residual + self.dropout(attn_output)
        
        # 2. 前馈网络分支
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + ff_output
        
        return x


class TimeSeriesEncoder(nn.Module):
    """
    基于RoPE的时序预测Encoder
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 配置参数
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.output_dim = getattr(config, 'output_dim', 1)
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # RoPE Transformer层
        self.encoder_layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=4 * self.hidden_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] 输入时序数据
        
        Returns:
            torch.Tensor: 预测输出
            - 如果output_dim=1: [batch_size] 每个样本一个预测值
            - 如果output_dim>1: [batch_size, output_dim] 每个样本多个预测值
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. 输入投影: [B, T, input_dim] -> [B, T, hidden_dim]
        x = self.input_projection(x)
        
        # 2. 通过所有RoPE Transformer层
        # 每层内部自动应用RoPE位置编码和因果掩码
        for layer in self.encoder_layers:
            x = layer(x)
        
        # 3. 取最后一个时间步的表示用于预测
        # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
        last_hidden = x[:, -1, :]
        
        # 4. 输出投影
        output = self.output_projection(last_hidden)
        
        # 5. 压缩最后一维
        output = output.squeeze(-1)  # [batch_size, 1] -> [batch_size]
        
        return output


# 向后兼容性别名
Encoder = TimeSeriesEncoder

# 配置类  
class EncoderConfig:
    def __init__(self, 
                 input_dim=133,
                 hidden_dim=128, 
                 num_layers=3,
                 num_heads=8,
                 dropout=0.1,
                 output_dim=1,
                 device='cuda:0'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.output_dim = output_dim
        self.device = device
        
        # 验证配置
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) 必须能被 num_heads ({num_heads}) 整除"


if __name__ == '__main__':
    # 设置随机种子保证可重现
    torch.manual_seed(42)
    
    # 测试配置
    config = EncoderConfig(
        input_dim=133,    # 输入特征维度
        hidden_dim=128,   # 隐藏层维度 (必须能被num_heads整除)
        num_layers=3,     # Transformer层数
        num_heads=8,      # 注意力头数
        dropout=0.1,      # Dropout率
        output_dim=1,
        device='cuda:0'
    )
    
    # 添加项目根目录到Python路径
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
    from get_data.CCB.CCB_dataloader import get_CCB_TimeSeriesDataloader
    TimeSeries_trainloader, TimeSeries_valiloader, TimeSeries_testloader = get_CCB_TimeSeriesDataloader(
        train_time_period='201901-202312', 
        test_time_period='202401-202504',
        shuffle_time=False,
        window_size=30,
        config=config
    )
    x = next(iter(TimeSeries_trainloader))[0]
    x = torch.nan_to_num(x, nan=0)
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    # batch_size = 32
    # seq_len = 30
    # x = torch.randn(batch_size, seq_len, config.input_dim)
    print(f"数据形状: {x.shape}")

    model = TimeSeriesEncoder(config).to(config.device)
    model.eval()
    
    print(f"\n📈 前向传播测试:")
    print(f"   - 输入形状: {x.shape}")
    
    start_time = time.time()
    with torch.no_grad():
        output = model(x.to(config.device))
    end_time = time.time()
    print(f"   - 前向传播时间: {end_time - start_time:.4f}秒")
    
    print(f"   - 输出形状: {output.shape}")
    print(f"   - 输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"   - 输出示例: {output[:5].tolist()}")
    
    # 验证形状
    expected_shape = (batch_size,) if config.output_dim == 1 else (batch_size, config.output_dim)
    assert output.shape == expected_shape, f"输出形状错误! 期望: {expected_shape}, 实际: {output.shape}"
    

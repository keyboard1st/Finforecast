import torch.nn as nn
import torch
import os

class GRU(nn.Module):
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
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=False,  # 输入形状为 [seq_len, batch, input_size]
            dropout=config.dropout
        )
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)
        self.model_name = 'GRU'

    def forward(self, x):
        # x 形状: [batch_size, seq_len, input_dim]
        # 转换为GRU需要的形状: [seq_len, batch_size, input_dim]
        x = x.permute(1, 0, 2)
        # GRU输出:
        # - output: 所有时间步的隐藏状态 [seq_len, batch, hidden_dim]
        # - h_n: 最后一个时间步的隐藏状态 [num_layers, batch, hidden_dim]
        output, h_n = self.gru(x)
        # last_hidden =  torch.cat((h_n[-2, :, :], h_n[-1, : , : ]), dim=-1)
        # 取最后一层的最终隐藏状态（若多层GRU）
        last_hidden = h_n[-1, :, :]  # 形状 [batch_size, hidden_dim]
        # 全连接层预测收益率
        pred = self.fc(last_hidden)  # 形状 [batch_size, output_dim]
        return pred.squeeze()

class two_GRU(nn.Module):
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
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=False,  # 输入形状为 [seq_len, batch, input_size]
            dropout=config.dropout
        )
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim//2, config.output_dim)
        self.model_name = 'two_GRU'

    def forward(self, x):
        # x 形状: [batch_size, seq_len, input_dim]
        # 转换为GRU需要的形状: [seq_len, batch_size, input_dim]
        x = x.permute(1, 0, 2)
        # GRU输出:
        # - output: 所有时间步的隐藏状态 [seq_len, batch, hidden_dim]
        # - h_n: 最后一个时间步的隐藏状态 [num_layers, batch, hidden_dim]
        output, h_n = self.gru(x)
        # last_hidden =  torch.cat((h_n[-2, :, :], h_n[-1, : , : ]), dim=-1)
        # 取最后一层的最终隐藏状态（若多层GRU）
        last_hidden = h_n[-1, :, :]  # 形状 [batch_size, hidden_dim]
        # 全连接层预测收益率
        last_hidden = self.relu(self.fc1(last_hidden))
        pred = self.fc2(last_hidden)  # 形状 [batch_size, output_dim]
        return pred.squeeze()



class BiGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=False,
            dropout=config.dropout,
            bidirectional=True  # 启用双向结构
        )

        # 调整全连接层输入维度
        self.fc = nn.Linear(config.hidden_dim * 2, config.output_dim)
        self.model_name = 'BiGRU'
    def forward(self, x):
        # 输入形状处理 [batch, seq, feature] -> [seq, batch, feature]
        x = x.permute(1, 0, 2)

        # 双向GRU输出处理
        output, h_n = self.gru(x)  # output形状 [seq_len, batch, hidden_dim*2]

        # 关键修改4：正确提取双向最终状态
        # h_n形状 [num_layers*2, batch, hidden_dim]
        last_hidden = output[-1, :, :]
        # [batch, hidden_dim * 2]

        # 全连接预测
        pred = self.fc(last_hidden)
        return pred.squeeze()


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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.input_dim,
            nhead= 2,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=False
        )
        self.bn_hidden = nn.BatchNorm1d(num_features=config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)

        output = self.encoder_layer(x)

        last_hidden = self.bn_hidden(output)

        # 全连接层预测收益率
        pred = self.fc(last_hidden)  # 形状 [batch_size, output_dim]
        return pred.squeeze()


def freeze_gru(model):
    """冻结除了最后一层的其他参数"""
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True
    return model


if __name__=='__main__':
    # exp_path = '/home/hongkou/chenx/exp/exp_006'
    # GRU_config = Config.from_json(os.path.join(exp_path, 'config.json'))
    #
    # model = GRU(GRU_config)
    # model = freeze_gru(model)
    # loss_f = nn.MSELoss()
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=GRU_config.learning_rate)
    # for _ in range(2):
    #     X = torch.randn(2, 5, 68)
    #     y = torch.randn(2)
    #     optimizer.zero_grad()
    #     pred = model(X)
    #     loss = loss_f(pred, y)
    #     loss.backward()
    #     original_gru_weights = [p.data.clone() for p in model.gru.parameters()]
    #     original_gru_weights_grad = [p.grad for p in model.gru.parameters()]
    #     original_fc_weights = [p.data.clone() for p in model.fc.parameters()]
    #     original_fc_weights_grad = [p.grad for p in model.fc.parameters()]
    #     print(f"original_gru_weights：", original_gru_weights[0][0])
    #     print(f"original_fc_weights：", original_fc_weights[1])
    #     print(f"original_gru_grad：", original_gru_weights_grad)
    #     print(f"original_fc_grad：", original_fc_weights_grad[1])
    #     print(loss.item())
    #     optimizer.step()

    torch.manual_seed(42)

    # 测试参数
    batch_size = 3280
    seq_len = 30
    input_dim = 312
    hidden_dim = 128
    output_dim = 1
    num_layers = 3
    dropout = 0.2
    class config:
        input_dim = input_dim
        hidden_dim = hidden_dim
        num_layers = num_layers
        output_dim = output_dim
        dropout = dropout


    # 初始化模型
    model = two_GRU(config)

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
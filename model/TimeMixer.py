import psutil
import torch
import torch.nn as nn

from model.layers.Autoformer_EncDec import series_decomp
from model.layers.Embed import DataEmbedding_wo_pos
from model.layers.StandardNorm import Normalize

process = psutil.Process()

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )

    def forward(self, x_list):
        # x_list: [[B * C, T, H], [B * C, T/2, H], [B * C, T/4, H]...]
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))
        # seasonal_list: [[B * C,H,T], [B * C,H,T/2], [B * C,H,T/4]...]
        # trend_list: [[B * C,H,T], [B * C,H,T/2], [B * C,H,T/4]...]

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # out_season_list: [[B * C, T, H], [B * C, T/2, H], [B * C, T/4, H]...]
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        # out_trend_list: [[B * C, T, H], [B * C, T/2, H], [B * C, T/4, H]...]

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):

    def __init__(self, configs):
        super(TimeMixer, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = 1
        self.down_sampling_window = configs.down_sampling_window
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    self.pred_len,
                )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        self.projection_layer = nn.Linear(
                configs.d_model, 1, bias=True)

        self.fc = nn.Sequential(
            nn.Linear(in_features=configs.enc_in, out_features=configs.enc_in//2),
            nn.GELU(),
            nn.Linear(in_features=configs.enc_in//2, out_features=1),
        )
        self.model_name = 'TimeMixer'

    def __multi_scale_process_inputs(self, x_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        x_enc = x_enc_sampling_list
        return x_enc



    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        x_list = x_list
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)


        return dec_out_list


    def forward(self, x_enc):
        # mem = []
        #
        # def log(name):
        #     # GPU 已分配显存 (MiB)
        #     gpu_mem = torch.cuda.memory_allocated() / 1024 ** 2
        #
        #     # CPU 物理内存占用 (RSS, MiB)
        #     cpu_mem = process.memory_info().rss / 1024 ** 2
        #
        #     mem.append((name, gpu_mem, cpu_mem))

        # log("start")
        # x_enc: [B,T,C]
        x_enc = self.__multi_scale_process_inputs(x_enc)
        # x_enc: [[B,T,C], [B,T/2,C], [B,T/4,C]...]
        # log("after multi_scale_process_inputs")

        x_list = []
        for i, x in enumerate(x_enc):
            B, T, C = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * C, T, 1)
            x_list.append(x)
        # log("after enc_embedding")
        # x_list: [[B * C, T, 1], [B * C, T/2, 1], [B * C, T/4, 1]...]


        # embedding
        enc_out_list = []
        for i, x in enumerate(x_list):
            # 2d卷积embedding
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            # log(f"after enc_embedding [{i}]")
            enc_out_list.append(enc_out)
        # enc_out_list: [[B * C, T, H], [B * C, T/2, H], [B * C, T/4, H]...]

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
            # log(f"after pdm_blocks[{i}]")
        # enc_out_list: [[B * C, T, H], [B * C, T/2, H], [B * C, T/4, H]...]

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        # log("after future_multi_mixing")

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        # dec_out: [B, 1, C]

        dec_out = self.fc(dec_out)
        # log("after fc")
        # for name, gpu, cpu in mem:
        #     print(f"{name:30s} | GPU: {gpu:8.1f} MiB | CPU: {cpu:8.1f} MiB")

        return dec_out.squeeze()


if __name__=='__main__':

    torch.manual_seed(42)

    # 测试参数
    batch_size = 3773
    seq_len = 96
    enc_in = 129
    class config:
        seq_len = seq_len
        enc_in = enc_in
        dec_in = enc_in
        c_out = enc_in
        down_sampling_layers = 3
        down_sampling_window = 2
        e_layers = 3    # pbm_layers
        decomp_method = 'moving_avg'
        down_sampling_method = 'avg'
        moving_avg = 25
        pred_len = 1
        d_model = 16
        d_ff = 32
        output_dim = 1
        dropout = 0.2
        use_norm = 0


    # 初始化模型
    model = TimeMixer(config)

    # 创建随机输入数据
    x = torch.randn(batch_size, seq_len, enc_in) # 形状 [batch_size, seq_len, input_dim]

    # 切换模型到评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        output = model(x)

    # 打印输出形状和部分输出值
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例 (前5个): {output[:5]}")




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class FactorValidator:
    def __init__(self, factor, 
                 manual_check=None,  # signature: (date: int, assets: List[int]) -> pd.Series
                 sample_n: int = 10,
                 plot_path: str = None, label: pd.DataFrame = None):
        """
        :param factor: DataFrame，index 为日期（int），columns 为资产代码（int）
        :param manual_check: 手动验证函数，输入（date, assets_list），返回对应 pd.Series
        :param sample_n: 最后一天抽样资产数量
        :param plot_path: 若不为 None，则保存分布直方图到此目录
        """
        self.factor = factor
        self.manual_check = manual_check
        self.sample_n = sample_n
        self.plot_path = plot_path
        self.label = label

    def validate(self):
        # 1. 计算因子
        f = self.factor
        print(">>> 1. 格式检查")
        assert isinstance(f, pd.DataFrame), "返回值必须是 DataFrame"
        assert pd.api.types.is_integer_dtype(f.index), "索引需为 int（8位日期）"
        assert pd.api.types.is_integer_dtype(f.columns), "列标签需为 int（标的代码）"
        print(f"✔ 返回 DataFrame，shape={f.shape}\n")

        # 2. 缺失值统计
        print(">>> 2. 缺失值统计")
        total_nan = f.isna().sum().sum()
        print(f"- 缺失值比例：{total_nan/f.size}")
        print("- 每个日期缺失值：")
        print(f.isna().sum(axis=1).describe())
        print("- 每个标的缺失值：")
        print(f.isna().sum(axis=0).describe(), "\n")


        # 3. 分布情况
        print(">>> 4. 描述性统计")
        print(f.describe().T, "\n")
        flat = f.values.flatten()
        if self.plot_path:
            os.makedirs(self.plot_path, exist_ok=True)
            plt.figure()
            plt.hist(flat[~np.isnan(flat)], bins=100)
            plt.title("Factor Distribution")
            plt.xlabel("Factor Value"); plt.ylabel("frequency")
            fig_path = os.path.join(self.plot_path, "histogram.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"✔ 已保存直方图到 {fig_path}\n")


        # 4. IC计算
        if self.label is not None:
            print(">>> 4. IC计算")
            IC = f.T.corrwith(self.label.T)
            N = IC.dropna().shape[0]
            mean_ic = IC.mean()
            std_ic  = IC.std(ddof=1)
            ir = mean_ic / std_ic if std_ic != 0 else np.nan
            print(f"有效样本数: {N}, 截面IC均值: {mean_ic:.4f}, IR: {ir:.4f}")

        # 5. 手动验证
        if self.manual_check:
            print(">>> 5. 手动验证（最后一天前10个标的）")
            last_date = f.index.max()
            last_date_notnan = f.loc[last_date, f.loc[last_date].notna()].index.tolist()
            assets = list(last_date_notnan[-self.sample_n:])
            auto_vals = f.loc[last_date, assets]
            try:
                manual_vals = self.manual_check(last_date, assets)
                compare = pd.DataFrame({
                    "自动计算": auto_vals,
                    "手动计算": manual_vals
                })
                compare["差值"] = compare["自动计算"] - compare["手动计算"]
                print(compare)
                max_diff = compare["差值"].abs().max()
                print(f"\n最大差值：{max_diff}")
            except Exception as e:
                print(f"❌ 手动验证出错：{e}")

        return f

if __name__ == "__main__":
    from day_factor_zoo import F1
    from get_data_from_pq import get_minute_stock, get_minute_ccb, get_daily_features
    label = pd.read_parquet(r'D:\chenxing\Finforecast\factor_warehouse\label\label_vwap_log_cliped')
    print(label.shape)
    def manual_F1(date, assets):
    # 取最后一分钟
        sc = get_minute_stock(date, 'close').iloc[-1]
        cc = get_minute_ccb(date, 'close').iloc[-1]
        swap = get_daily_features('swap_share_price').loc[date, assets]
        ratio = cc[assets] / sc[assets]
        return ratio/100 * swap - 1

    validator = FactorValidator(
        factor_func=F1,
        manual_check=manual_F1,
        sample_n=10,
        plot_path=r"D:\chenxing\Finforecast\factor_warehouse\plots\F1",
        label=label
    )
    f1_df = validator.validate()

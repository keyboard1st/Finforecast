import itertools
import os
from copy import deepcopy

import pandas as pd

from train.GBDT_trainer import tree_test, tree_pred
from train import GRU_pred_market_new, GRU_fin_test_new


def test_all_model(PathConfig, TimeSeries_xy_loader, CrossSection_xy_loader, Minute_xy_loader = None, **models):
    model_pred_dict = {}  # 用来存各个模型的 pred_arr
    for name, model in models.items():
        if name == 'GRU':
            print('----------------------GRU test--------------------------')
            GRU_pred_arr, GRU_true = GRU_fin_test_new(TimeSeries_xy_loader, model)
            print("GRU pred shape ", GRU_pred_arr.shape)
            print("GRU true shape ", GRU_true.shape)
            model_pred_dict[name] = GRU_pred_arr
            model_pred_dict['label'] = GRU_true
        elif name == 'TimeMixer' and PathConfig.use_minute_model:
            print('----------------------TimeMixer test-----------------------')
            Minute_pred_arr, Minute_true = GRU_fin_test_new(Minute_xy_loader, model)
            print("Minute pred shape ", Minute_pred_arr.shape)
            print("Minute true shape ", Minute_true.shape)
            model_pred_dict[name] = Minute_pred_arr
        else:
            print(f'----------------------{name} test-----------------------')
            cur_loader = deepcopy(CrossSection_xy_loader)
            tree_pred_arr, tree_true = tree_test(cur_loader, model)
            print(f"{name} pred shape ", tree_pred_arr.shape)
            print(f"{name} true shape ", tree_true.shape)
            model_pred_dict[name] = tree_pred_arr

    return model_pred_dict

def pred_all_model(PathConfig, TimeSeries_x_loader, CrossSection_x_loader, index_and_clos, Minute_x_loader = None, **models):
    model_pred_df_dict = {}
    for name, model in models.items():
        if name == 'GRU':
            print('----------------------GRU Pred--------------------------')
            GRU_pred_arr = GRU_pred_market_new(TimeSeries_x_loader, model)
            print("GRU pred shape ", GRU_pred_arr.shape)
            GRU_fin_pred_df = pd.DataFrame(GRU_pred_arr)
            GRU_fin_pred_df.index = index_and_clos.index
            GRU_fin_pred_df.columns = index_and_clos.columns
            model_pred_df_dict[name] = GRU_fin_pred_df
        elif name == 'TimeMixer' and PathConfig.use_minute_model:
            print('----------------------TimeMixer Pred-----------------------')
            Minute_pred_arr = GRU_pred_market_new(Minute_x_loader, model)
            print("Minute pred shape ", Minute_pred_arr.shape)
            Minute_fin_pred_df = pd.DataFrame(Minute_pred_arr)
            Minute_fin_pred_df.index = index_and_clos.index
            Minute_fin_pred_df.columns = index_and_clos.columns
            model_pred_df_dict[name] = Minute_fin_pred_df
        else:
            print(f'----------------------{name} Pred-----------------------')
            cur_loader = deepcopy(CrossSection_x_loader)
            tree_pred_arr = tree_pred(cur_loader, model)
            print(f"{name} pred shape ", tree_pred_arr.shape)
            tree_pred_df = pd.DataFrame(tree_pred_arr)
            tree_pred_df.index = index_and_clos.index
            tree_pred_df.columns = index_and_clos.columns
            model_pred_df_dict[name] = tree_pred_df

    model_pred_df_dict['GBDT'] = (model_pred_df_dict['XGBoost'] + model_pred_df_dict['LightGBM'] + model_pred_df_dict['CatBoost']) / 3
    model_pred_df_dict.pop('XGBoost',None)
    model_pred_df_dict.pop('LightGBM',None)
    model_pred_df_dict.pop('CatBoost',None)

    return model_pred_df_dict

def get_weights(n):
    '''
    返回将1拆成n个加和为1的权重列表，每个权重为0.1的倍数，例如n=3时：
    [[0.1, 0.1, 0.8], [0.1, 0.2, 0.7], ...]
    '''
    steps = [i / 10 for i in range(1, 10)]  # 0.1 到 0.9（不包含0）
    weights = []
    for w in itertools.product(steps, repeat=n):
        if abs(sum(w) - 1.0) < 1e-8:
            weights.append(list(w))
    return weights

def model_mixer(PathConfig, model_pred_df_dict, labels_df, market_cap_df=None):
    print('----------------------model mixer-----------------------')
    print('base models:', model_pred_df_dict.keys())
    if PathConfig.use_minute_model:
        avg_pred = (model_pred_df_dict['GRU'] + model_pred_df_dict['TimeMixer'] + model_pred_df_dict['GBDT']) / 3.0
        model_pred_df_dict['0.33GRU & 0.33TimeMixer & 0.33GBDT'] = avg_pred
        weights = get_weights(3)
        ic_records = []
        for w1, w2, w3 in weights:
            # 加权预测
            mix_pred = (
                    w1 * model_pred_df_dict['GRU'] +
                    w2 * model_pred_df_dict['TimeMixer'] +
                    w3 * model_pred_df_dict['GBDT']
            )
            # 只在市值>0的位置计算 IC
            if market_cap_df is not None:
                masked = mix_pred.where(market_cap_df > 0)
            else:
                masked = mix_pred
            # 按列（股票）计算与 labels 的相关，再取平均
            ic = (masked.T.corrwith(labels_df.T)).mean()
            ic_records.append(((w1, w2, w3), ic, mix_pred))
            print(f'{w1}GRU & {w2}TimeMixer & {w3}GBDT, IC = {ic:.4f}')

        top5 = sorted(ic_records, key=lambda x: x[1], reverse=True)[:5]
        for idx, ((w1, w2, w3), ic, pred_df) in enumerate(top5, 1):
            name = f'{w1}GRU & {w2}TimeMixer & {w3}GBDT'
            print(f'top{idx} {name}, IC = {ic:.4f}')
            model_pred_df_dict[name] = pred_df
        print('final models:', list(model_pred_df_dict.keys()))

    else:
        weights = get_weights(2)
        ic_records = []
        for w1, w2 in weights:
            mix_pred = (w1 * model_pred_df_dict['GRU'] + w2 * model_pred_df_dict['GBDT'])
            # 只在市值>0的位置计算 IC
            if market_cap_df is not None:
                masked = mix_pred.where(market_cap_df > 0)
            else:
                masked = mix_pred
            # 按列（股票）计算与 labels 的相关，再取平均
            ic = (masked.T.corrwith(labels_df.T)).mean()
            ic_records.append(((w1, w2), ic, mix_pred))
            print(f'{w1}GRU & {w2}GBDT, IC = {ic:.4f}')
        top5 = sorted(ic_records, key=lambda x: x[1], reverse=True)[:5]
        for idx, ((w1, w2), ic, pred_df) in enumerate(top5, 1):
            name = f'{w1}GRU & {w2}GBDT'
            print(f'top{idx} {name}, IC = {ic:.4f}')
            model_pred_df_dict[name] = pred_df
        print('final models:', list(model_pred_df_dict.keys()))


    return model_pred_df_dict

def save_pred(PathConfig, model_pred_df_dict, labels_df, market_cap_df=None):
    model_pred_df_dict_with_ic = {}
    for name, pred_df in model_pred_df_dict.items():
        if market_cap_df is not None:
            pred_df_mask = pred_df.where(market_cap_df>0)
        else:
            pred_df_mask = pred_df
        pred_df_mask.to_csv(os.path.join(PathConfig.save_path, f'{name}_fin_pred_mask.csv'))
        pred_ic = (pred_df_mask.T.corrwith(labels_df.T)).mean()
        print(f'{name} Pred ic: {pred_ic:.4f}')
        new_name = name + str(format(pred_ic, '.4f'))
        model_pred_df_dict_with_ic[new_name] = pred_df_mask
    return model_pred_df_dict_with_ic
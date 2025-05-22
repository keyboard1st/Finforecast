from train.GBDT_trainer import lgb_train_and_test, xgb_train_and_test, cat_train_and_test
from metrics.calculate_ic import ic_between_arr
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from config import get_config
config = get_config()

def lgb_xgb_cat_train_and_save(trainloader, testloader, exp_path):
    lgb_trainloader, lgb_testloader = deepcopy(trainloader), deepcopy(testloader)
    xgb_trainloader, xgb_testloader = deepcopy(trainloader), deepcopy(testloader)
    cat_trainloader, cat_testloader = trainloader, testloader
    lgb_pred_arr, lgb_ture = lgb_train_and_test(lgb_trainloader, lgb_testloader, exp_path)
    lgb_ic = ic_between_arr(lgb_pred_arr, lgb_ture)
    print(f"LightGBM test IC: {lgb_ic:.4f}")
    xgb_pred_arr, xgb_true = xgb_train_and_test(xgb_trainloader, xgb_testloader, exp_path)
    xgb_ic = ic_between_arr(xgb_pred_arr, xgb_true)
    print(f"xgBoost test IC: {xgb_ic:.4f}")
    cat_pred_arr, cat_true = cat_train_and_test(cat_trainloader, cat_testloader, exp_path)
    cat_ic = ic_between_arr(cat_pred_arr, cat_true)
    print(f"catBoost test IC: {cat_ic:.4f}")

    assert xgb_pred_arr.shape == lgb_pred_arr.shape == cat_pred_arr.shape
    gbdt_pred_arr = (xgb_pred_arr + cat_pred_arr + lgb_pred_arr) / 3
    gbdt_ic = ic_between_arr(gbdt_pred_arr, lgb_ture)
    print(f"gbdt test IC: {gbdt_ic:.4f}")


if __name__ == '__main__':
    from get_data.CY312.CY312_rollingtrain_dataloader import get_CY312_rollingtrain_CrossSectionLoader

    CrossSection_trainloader, CrossSection_testloader = get_CY312_rollingtrain_CrossSectionLoader(batchsize="all",
                                                                                                  shuffle_time=False,
                                                                                                  time_period=config.time_period)
    config.exp_path = f'/home/hongkou/TimeSeries/exp/{config.task_name}'
    lgb_xgb_cat_train_and_save(CrossSection_trainloader, CrossSection_testloader, config.exp_path)



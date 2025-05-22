from config import get_config
config = get_config()

def get_loader(loader_type:str, fintest:bool):
    assert loader_type in ['CrossSection', 'TimeSeries']
    if loader_type == 'CrossSection':
        if config.factor_name == 'Ding128':
            if config.train_type == 'last_year_train':
                if fintest:
                    from Ding128.Ding128_CrossSection_dataloader import get_Ding128_CrossSectionFintestloader
                    fintestloader = get_Ding128_CrossSectionFintestloader()
                    return fintestloader
                else:
                    from Ding128.Ding128_CrossSection_dataloader import get_Ding128_CrossSectionLoader
                    train_dataloader, test_dataloader = get_Ding128_CrossSectionLoader('all',False)
                    return train_dataloader, None, test_dataloader
            else:
                raise ValueError("Ding128 has no rollingtrain data")
        elif config.factor_name == 'CY312':
            if config.train_type == 'last_year_train':
                if fintest:
                    raise ValueError("CY312 last_year_train has no fintest data")
                else:
                    from CY312.CY312_CrossSection_dataloader import get_CY312_CrossSectionDataloader
                    train_dataloader,test_dataloader = get_CY312_CrossSectionDataloader(batchsize='all',shuffle_time=False)
                    return train_dataloader, None, test_dataloader
            elif config.train_type == 'rolling_train':
                if fintest:
                    from CY312.CY312_rollingtrain_dataloader import get_CY312_rollingfintest_CrossSectionLoader
                    fintestloader = get_CY312_rollingfintest_CrossSectionLoader(time_period = config.time_period)
                    return fintestloader
                else:
                    from CY312.CY312_rollingtrain_dataloader import get_CY312_rollingtrain_CrossSectionLoader
                    train_dataloader,test_dataloader = get_CY312_rollingtrain_CrossSectionLoader(batchsize="all", shuffle_time=False, time_period=config.time_period)
                    return train_dataloader, None, test_dataloader
        elif config.factor_name == 'DrJin129':
            if config.train_type == 'last_year_train':
                if fintest:
                    raise ValueError("DrJin129 last_year_train has no fintest data")
                else:
                    from DrJin129.DrJin129_ori_dataloader import get_DrJin129_CrossSectionDatasetLoader
                    train_dataloader,test_dataloader = get_DrJin129_CrossSectionDatasetLoader(batchsize="all", shuffle_time=False)
                    return train_dataloader, None, test_dataloader

            elif config.train_type == 'rolling_train':
                if fintest:
                    from DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingfintest_CrossSectionDatasetLoader
                    fintestloader = get_DrJin129_rollingfintest_CrossSectionDatasetLoader()
                    return fintestloader
                else:
                    from DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingtrain_CrossSectionDatasetLoader
                    train_dataloader,test_dataloader = get_DrJin129_rollingtrain_CrossSectionDatasetLoader(batchsize="all", shuffle_time=False, time_period=config.time_period)
                    return train_dataloader, None, test_dataloader


    elif loader_type == 'TimeSeries':
        if config.factor_name == 'Ding128':
            if config.train_type == 'last_year_train':
                if fintest:
                    from Ding128.Ding128_TimeSeries_dataloader import get_Ding128_TimeSeriesFintestloader
                    fintestloader = get_Ding128_TimeSeriesFintestloader()
                    return fintestloader
                else:
                    from Ding128.Ding128_TimeSeries_dataloader import get_Ding128_TimeSeriesLoader
                    train_dataloader, val_dataloader, test_dataloader = get_Ding128_TimeSeriesLoader(batchsize = 1, shuffle_time = config.shuffle_time, window_size = config.window_size, num_val_windows = config.num_val_windows, val_sample_mode = 'random')
                    return train_dataloader, val_dataloader, test_dataloader
            else:
                raise ValueError("Ding128 has no rollingtrain data")

        elif config.factor_name == 'CY312':
            if config.train_type == 'last_year_train':
                if fintest:
                    raise ValueError("CY312 last_year_train has no fintest data")
                else:
                    from CY312.CY312_CrossSection_dataloader import get_CY312_CrossSectionDataloader
                    train_dataloader,test_dataloader = get_CY312_CrossSectionDataloader(batchsize='all',shuffle_time=False)
                    return train_dataloader, None, test_dataloader
            elif config.train_type == 'rolling_train':
                if fintest:
                    from CY312.CY312_rollingtrain_dataloader import get_CY312_rollingfintest_CrossSectionLoader
                    fintestloader = get_CY312_rollingfintest_CrossSectionLoader(time_period = config.time_period)
                    return fintestloader
                else:
                    from CY312.CY312_rollingtrain_dataloader import get_CY312_rollingtrain_CrossSectionLoader
                    train_dataloader,test_dataloader = get_CY312_rollingtrain_CrossSectionLoader(batchsize="all", shuffle_time=False, time_period=config.time_period)
                    return train_dataloader, None, test_dataloader
        elif config.factor_name == 'DrJin129':
            if config.train_type == 'last_year_train':
                if fintest:
                    raise ValueError("DrJin129 last_year_train has no fintest data")
                else:
                    from DrJin129.DrJin129_ori_dataloader import get_DrJin129_CrossSectionDatasetLoader
                    train_dataloader,test_dataloader = get_DrJin129_CrossSectionDatasetLoader(batchsize="all", shuffle_time=False)
                    return train_dataloader, None, test_dataloader

            elif config.train_type == 'rolling_train':
                if fintest:
                    from DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingfintest_CrossSectionDatasetLoader
                    fintestloader = get_DrJin129_rollingfintest_CrossSectionDatasetLoader()
                    return fintestloader
                else:
                    from DrJin129.DrJin129_rollingtrain_dataloader import get_DrJin129_rollingtrain_CrossSectionDatasetLoader
                    train_dataloader,test_dataloader = get_DrJin129_rollingtrain_CrossSectionDatasetLoader(batchsize="all", shuffle_time=False, time_period=config.time_period)
                    return train_dataloader, None, test_dataloader

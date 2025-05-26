from tqdm import tqdm
import time
import random
from collections import deque
import os
import torch
import numpy as np
import pandas as pd

from model.losses import IC_loss_double_diff
from metrics.calculate_ic import ic_between_timestep, ic_between_arr
from utils.fillna import filter_and_fillna, dropx_and_fillna
from metrics.train_plot import TrainingMetrics


def validate_one_epoch(val_dataloader, model, criterion):
    """
    在训练过程中的验证
    对x填0并且dropy中的nan值
    计算ic和loss
    """
    model.eval()
    total_loss = []
    ic_list = []

    with torch.inference_mode():
        for x, y in val_dataloader:
            x, y = x.float(), y.float()
            x, y = filter_and_fillna(x, y)
            outputs = model(x)
            true = y.detach()
            pred = outputs.detach()
            # 计算IC
            ic = ic_between_timestep(pred, true)
            if not np.isnan(ic):
                ic_list.append(ic)

            loss = criterion(pred, true)
            total_loss.append(loss.item())

    total_loss = np.average(total_loss)
    avg_ic = np.mean(ic_list) if ic_list else 0
    return total_loss, avg_ic

def train_one_epoch(config, epoch, model, train_dataloader, window_size, model_optim, criterion, scaler, scheduler, logger):
    '''
    :input:训练一轮所需参数
    :return: 训练一轮的loss
    '''
    model.train()
    train_loss = []

    for time_step, (batch_x, batch_y) in enumerate(train_dataloader):
        # print("train batch_x device", batch_x.device)
        model.train()
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        batch_x, batch_y_dropna = filter_and_fillna(batch_x, batch_y)
        # print("train batch_x device", batch_x.device)
        if batch_x.shape[0] == 0:
            print(f"{time_step} no valid x")
            continue

        model_optim.zero_grad()
        n_chunks = 4
        xb_chunks = torch.chunk(batch_x, n_chunks, dim=0)
        yb_chunks = torch.chunk(batch_y_dropna, n_chunks, dim=0)
        for xb_sub, y_sub in zip(xb_chunks, yb_chunks):
            # xb_sub.shape == [B_i, T, C], y_sub.shape == [B_i, ...]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(xb_sub)
                # print("out put device", out.device)
                loss = criterion(out, y_sub)
                scaler.scale(loss / n_chunks).backward()
        scaler.step(model_optim)
        scaler.update()

        train_loss.append(loss.item())


        if (time_step + 1) % 100 == 0:
            logger.info(
                "\ttime_step: {0}, epoch: {1} | train_loss: {2:.7f}".format(
                    time_step + 1, epoch + 1, loss.item()))

        if config.lradj == 'TST':
            scheduler.step()
    if train_loss:
        epoch_train_loss = np.mean(train_loss)
    else:
        raise ValueError("No training loss")

    return epoch_train_loss, model

def train_one_epoch_and_validate(config, epoch, model, train_dataloader, window_size, idx, cache, model_optim, criterion, scaler, scheduler, logger):
    '''
    :input:训练一轮所需参数
    :return: 训练一轮的loss
    '''
    model.train()
    train_loss = []
    cross_val_dataloader = iter(train_dataloader)
    for _ in range(window_size - 1):
        next(cross_val_dataloader)
    for time_step, (batch_x, batch_y) in enumerate(train_dataloader):
        # print("train batch_x device", batch_x.device)
        model.train()
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        if time_step in idx:
            sampled_x = batch_x[:, -1, :]
            cache.append(sampled_x)
        batch_x, batch_y_dropna = filter_and_fillna(batch_x, batch_y)
        # print("train batch_x device", batch_x.device)
        if batch_x.shape[0] == 0:
            print(f"{time_step} no valid x")
            continue

        model_optim.zero_grad()
        n_chunks = 4
        xb_chunks = torch.chunk(batch_x, n_chunks, dim=0)
        yb_chunks = torch.chunk(batch_y_dropna, n_chunks, dim=0)
        for xb_sub, y_sub in zip(xb_chunks, yb_chunks):
            # xb_sub.shape == [B_i, T, C], y_sub.shape == [B_i, ...]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(xb_sub)
                # print("out put device", out.device)
                loss = criterion(out, y_sub)
                scaler.scale(loss / n_chunks).backward()
        scaler.step(model_optim)
        scaler.update()

        train_loss.append(loss.item())
        is_last_batch = (time_step >= (len(train_dataloader) - window_size))
        if not is_last_batch:
            batch_x_val, batch_y_val = next(cross_val_dataloader)
            val_x, val_y = batch_x_val.float(), batch_y_val.float()
        else:
            val_x = torch.stack(list(cache), dim=1)
            val_y = batch_y

        val_x, val_y = filter_and_fillna(val_x, val_y)
        model.eval()
        with torch.inference_mode():
            outputs = model(val_x)
            cval_loss = criterion(outputs, val_y)
            cval_loss = cval_loss.item()
            cval_ic = ic_between_timestep(outputs, val_y)

        if (time_step + 1) % 100 == 0:
            logger.info(
                "\ttime_step: {0}, epoch: {1} | train_loss: {2:.7f} | cval_loss: {3:.7f} | cval_ic: {4:.4f}".format(
                    time_step + 1, epoch + 1, loss.item(), cval_loss, cval_ic))

        if config.lradj == 'TST':
            scheduler.step()
    if train_loss:
        epoch_train_loss = np.mean(train_loss)
    else:
        raise ValueError("No training loss")

    return epoch_train_loss, model

def GRU_fin_test(test_loader, model):
    """
    模型测试阶段，只需要计算IC，不计算Loss
    对x进行填充0处理
    """
    model.eval()
    pred_list = []
    true_list = []
    # with torch.no_grad():
    with torch.inference_mode():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.float(), y.float()
            x = torch.nan_to_num(x, nan=0)
            pred = model(x)
            pred_cpu = pred.squeeze().detach().cpu().numpy()  # 先detach再转CPU
            y_cpu = y.squeeze().detach().cpu().numpy()
            true_list.append(y_cpu)
            pred_list.append(pred_cpu)
    true_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in true_list], axis=0)
    pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)

    return pred_arr, true_arr

def GRU_pred_market(mkt_align_x_loader, model):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for x in tqdm(mkt_align_x_loader, desc="Prediction"):
            x = x.float()
            x = torch.nan_to_num(x, nan=0)
            pred = model(x)
            pred_cpu = pred.squeeze().detach().cpu().numpy()
            pred_list.append(pred_cpu)
    pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)
    return pred_arr

# 普通训练
def norm_train(config, train_dataloader, val_dataloader, test_dataloader, model, early_stopping, model_optim, criterion, scheduler, logger, scaler):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    # print("train",device)
    window_size, lradj, exp_path, train_epochs = config.window_size, config.lradj, config.exp_path, config.train_epochs

    if not os.path.exists(os.path.join(exp_path, 'models/')):
        os.makedirs(os.path.join(exp_path, 'models/'))
    # model and loss
    model = model.to(device)
    # print("model.device:", next(model.parameters()).device)
    criterion = criterion.to(device)

    # metrics
    best_ic = 0
    metrics = TrainingMetrics(exp_path)

    # ---------train epochs-------------
    for epoch in range(train_epochs):
        epoch_time = time.time()

        # -----------train one epoch and validate---------------
        epoch_train_loss, model = train_one_epoch(config, epoch, model, train_dataloader, window_size, model_optim, criterion, scaler, scheduler, logger)

        vali_loss, vali_ic_mean = validate_one_epoch(val_dataloader, model, criterion)
        test_loss, test_ic_mean = validate_one_epoch(test_dataloader, model, criterion)

        logger.info("Epoch: {} | Time: {:.2f}s".format(epoch, time.time() - epoch_time))
        logger.info("Train Loss: {:.4f} | Val Loss: {:.4f} | Test Loss: {:.4f} | Val IC Mean: {:.4f}| Test IC Mean: {:.4f}".format(epoch_train_loss, vali_loss, test_loss, vali_ic_mean, test_ic_mean))

        early_model_path = os.path.join(exp_path, 'models/early_model.pth')
        early_stopping(vali_loss, model, early_model_path)
        if early_stopping.early_stop:
            logger.info("-----------Early stopping----------")
            break

        if lradj == 'TST':
            logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        elif lradj == 'cos':
            scheduler.step(epoch)
            logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        metrics.add_metrics(
            epoch=epoch,
            train_loss=epoch_train_loss,
            val_loss=vali_loss,
            test_loss=test_loss,
            val_ic_mean=vali_ic_mean,
            test_ic_mean=test_ic_mean,
            lr=model_optim.param_groups[0]['lr']
        )

        best_model_path = os.path.join(exp_path, 'models/best_model.pth')
        if best_ic < test_ic_mean:
            best_ic = test_ic_mean
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"best model saved with test ic : {best_ic:.4f}")

    last_model_path = os.path.join(exp_path, 'models/last_model.pth')
    torch.save(model.state_dict(), last_model_path)
    logger.info(f"last model saved with test ic : {test_ic_mean:.4f}")
    metrics.plot_loss(prefix="final_")
    metrics.plot_ic(prefix="final_")
    metrics.plot_lr(prefix="final_")
    df_metrics = metrics.to_dataframe()
    return model, df_metrics

# 普通训练并滚动验证
def train_and_cross_time_train(config, train_dataloader, val_dataloader, test_dataloader, model, early_stopping, model_optim, criterion, scheduler, logger, scaler):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    # print("train",device)
    window_size, lradj, exp_path, train_epochs = config.window_size, config.lradj, config.exp_path, config.train_epochs

    if not os.path.exists(os.path.join(exp_path, 'models/')):
        os.makedirs(os.path.join(exp_path, 'models/'))
    # model and loss
    model = model.to(device)
    # print("model.device:", next(model.parameters()).device)
    criterion = criterion.to(device)

    # if cross train is True, use IC_loss_double_diff
    if config.cross_train:
        IC_loss = IC_loss_double_diff().to(device)

    # metrics
    best_ic = 0
    metrics = TrainingMetrics(exp_path)

    # cache for cross validation
    cache = deque(maxlen=window_size)
    total_time_steps = len(train_dataloader)
    idx = random.Random(42).sample(range(total_time_steps - window_size), window_size)

    # ---------train epochs-------------
    for epoch in range(train_epochs):
        epoch_time = time.time()

        # -----------train one epoch and validate---------------
        epoch_train_loss, model = train_one_epoch_and_validate(config, epoch, model, train_dataloader, window_size, idx, cache, model_optim, criterion, scaler, scheduler, logger)

        vali_loss, vali_ic_mean = validate_one_epoch(val_dataloader, model, criterion)
        test_loss, test_ic_mean = validate_one_epoch(test_dataloader, model, criterion)

        logger.info("Epoch: {} | Time: {:.2f}s".format(epoch, time.time() - epoch_time))
        logger.info("Train Loss: {:.4f} | Val Loss: {:.4f} | Test Loss: {:.4f} | Val IC Mean: {:.4f}| Test IC Mean: {:.4f}".format(epoch_train_loss, vali_loss, test_loss, vali_ic_mean, test_ic_mean))

        early_model_path = os.path.join(exp_path, 'models/early_model.pth')
        early_stopping(vali_loss, model, early_model_path)
        if early_stopping.early_stop:
            logger.info("-----------Early stopping----------")
            break

        if lradj == 'TST':
            logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        elif lradj == 'cos':
            scheduler.step(epoch)
            logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        metrics.add_metrics(
            epoch=epoch,
            train_loss=epoch_train_loss,
            val_loss=vali_loss,
            test_loss=test_loss,
            val_ic_mean=vali_ic_mean,
            test_ic_mean=test_ic_mean,
            lr=model_optim.param_groups[0]['lr']
        )

        best_model_path = os.path.join(exp_path, 'models/best_model.pth')
        if best_ic < test_ic_mean:
            best_ic = test_ic_mean
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"best model saved with test ic : {best_ic:.4f}")

    last_model_path = os.path.join(exp_path, 'models/last_model.pth')
    torch.save(model.state_dict(), last_model_path)
    logger.info(f"last model saved with test ic : {test_ic_mean:.4f}")
    metrics.plot_loss(prefix="final_")
    metrics.plot_ic(prefix="final_")
    metrics.plot_lr(prefix="final_")
    df_metrics = metrics.to_dataframe()
    return model, df_metrics

# 训练并二次IC训练(未完成)
def train_one_epoch_and_train_twice(epoch, model, train_dataloader, window_size, idx, cache, model_optim, criterion, scaler, scheduler, logger, IC_loss, lgb_model):
    model.train()
    train_loss = []
    val_loss = []
    val_ic = []
    cross_val_dataloader = iter(train_dataloader)
    for _ in range(window_size - 1):
        next(cross_val_dataloader)
    for time_step, (batch_x, batch_y) in enumerate(train_dataloader):
        model.train()
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if time_step in idx:
            sampled_x = batch_x[:, -1, :]
            cache.append(sampled_x)
        batch_x, batch_y_dropna = filter_and_fillna(batch_x, batch_y)
        if batch_x.shape[0] == 0:
            print(f"{time_step} no valid x")
            continue
        model_optim.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(batch_x)
            # pred_lgb = lgb_model.predict(batch_x[:, -1, :].cpu().detach().numpy())
            # loss = criterion(outputs.requires_grad_(True), batch_y_dropna.requires_grad_(False), torch.tensor(pred_lgb, requires_grad=False).to(device))
            loss = criterion(outputs, batch_y_dropna)
        # loss.backward(retain_graph=True)
        # loss.backward()
        # model_optim.step()
        scaler.scale(loss).backward()
        # print(f"\nEpoch {epoch} time_step {time_step} Gradients:")
        # total_grad_norm = 0
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.data.mean().item()
        #         total_grad_norm += grad_norm
        #         print(f"{name:30} Grad: {grad_norm:.6f}")
        #     else:
        #         print(f"{name:30} No gradient")
        scaler.step(model_optim)
        scaler.update()
        # if np.isnan(loss.item()):
        #     print(f"\n{time_step} train loss is nan")
        #     print(f"\nbatch_x.shape:{batch_x.shape}, batch_y_dropna shape:{batch_y_dropna.shape}, output shape:{outputs.shape}")
        #     if torch.isnan(batch_y_dropna).any() or torch.isinf(batch_y_dropna).any():
        #         print("batch_y_dropna 包含 nan 或 inf!")
        #
        #     if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        #         print("outputs 包含 nan 或 inf!")
        #
        #     if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
        #         print("batch_x 包含 nan 或 inf!")

        train_loss.append(loss.item())
        is_last_batch = (time_step >= (len(train_dataloader) - window_size))
        if not is_last_batch:
            batch_x_val, batch_y_val = next(cross_val_dataloader)
            val_x, val_y = batch_x_val.float().to(device), batch_y_val.float().to(device)
        else:
            val_x = torch.stack(list(cache), dim=1)
            val_y = batch_y
        if lgb_model is not None:
            logger.info("cross train")
            val_x, val_y = filter_and_fillna(val_x, val_y)
            model.train()
            # model_optim.zero_grad()
            outputs = model(val_x)
            pred_lgb = lgb_model.predict(val_x[:, -1, :].cpu().detach().numpy())
            cval_loss = IC_loss(torch.tensor(pred_lgb, requires_grad=True).to(device), outputs.requires_grad_(True),
                                val_y.requires_grad_(True))
            cval_loss.backward()
            model_optim.step()
            cval_loss = cval_loss.item()
            # print(outputs.devie, outputs.requires_grad())
            # cval_ic = ic_between_timestep(outputs, val_y)
            cval_ic = pd.Series(outputs.cpu().detach().numpy()).corr(pd.Series(val_y.cpu().detach().numpy()))
        else:
            val_x, val_y = filter_and_fillna(val_x, val_y)
            model.eval()
            # with torch.no_grad():
            with torch.inference_mode():
                outputs = model(val_x)
                cval_loss = criterion(outputs, val_y)
                cval_loss = cval_loss.item()
                cval_ic = ic_between_timestep(outputs, val_y)

        if cval_loss is not None:
            val_loss.append(cval_loss)
        if cval_ic is not None:
            val_ic.append(cval_ic)

        if (time_step + 1) % 100 == 0:
            logger.info(
                "\ttime_step: {0}, epoch: {1} | train_loss: {2:.7f} | cval_loss: {3:.7f} | cval_ic: {4:.4f}".format(
                    time_step + 1, epoch + 1, loss.item(), cval_loss, cval_ic))
            # logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

        if config.lradj == 'TST':
            scheduler.step()

def train_and_cross_time_train_old(train_dataloader, val_dataloader, test_dataloader, model, early_stopping, model_optim, criterion, train_epochs, scheduler, window_size, lradj, logger, exp_path, scaler, lgb_model=None):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    time_now = time.time()
    best_ic = 0
    criterion = criterion.to(device)
    IC_loss = IC_loss_double_diff().to(device)
    metrics = TrainingMetrics(exp_path)
    cache = deque(maxlen=window_size)
    total_time_steps = len(train_dataloader)
    idx = random.Random(42).sample(range(total_time_steps - window_size), window_size)
    for epoch in range(train_epochs):
        model.train()
        train_loss = []
        val_loss = []
        val_ic = []
        epoch_time = time.time()
        cross_val_dataloader = iter(train_dataloader)
        for _ in range(window_size - 1):
            next(cross_val_dataloader)
        for time_step, (batch_x, batch_y) in enumerate(train_dataloader):

            model.train()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if time_step in idx:
                sampled_x = batch_x[:, -1, :]
                cache.append(sampled_x)
            batch_x, batch_y_dropna = filter_and_fillna(batch_x, batch_y)
            if batch_x.shape[0] == 0:
                print(f"{time_step} no valid x")
                continue
            model_optim.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(batch_x)
            # pred_lgb = lgb_model.predict(batch_x[:, -1, :].cpu().detach().numpy())
            # loss = criterion(outputs.requires_grad_(True), batch_y_dropna.requires_grad_(False), torch.tensor(pred_lgb, requires_grad=False).to(device))
                loss = criterion(outputs, batch_y_dropna)
            # loss.backward(retain_graph=True)
            # loss.backward()
            #
            # model_optim.step()
            scaler.scale(loss).backward()
            # print(f"\nEpoch {epoch} time_step {time_step} Gradients:")
            # total_grad_norm = 0
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.mean().item()
            #         total_grad_norm += grad_norm
            #         print(f"{name:30} Grad: {grad_norm:.6f}")
            #     else:
            #         print(f"{name:30} No gradient")
            scaler.step(model_optim)
            scaler.update()
            # if np.isnan(loss.item()):
            #     print(f"\n{time_step} train loss is nan")
            #     print(f"\nbatch_x.shape:{batch_x.shape}, batch_y_dropna shape:{batch_y_dropna.shape}, output shape:{outputs.shape}")
            #     if torch.isnan(batch_y_dropna).any() or torch.isinf(batch_y_dropna).any():
            #         print("batch_y_dropna 包含 nan 或 inf!")
            #
            #     if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            #         print("outputs 包含 nan 或 inf!")
            #
            #     if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
            #         print("batch_x 包含 nan 或 inf!")

            train_loss.append(loss.item())
            is_last_batch = (time_step >= (len(train_dataloader) - window_size))
            if not is_last_batch:
                batch_x_val, batch_y_val = next(cross_val_dataloader)
                val_x, val_y = batch_x_val.float().to(device), batch_y_val.float().to(device)
            else:
                val_x = torch.stack(list(cache), dim=1)
                val_y = batch_y
            if lgb_model is not None:
                logger.info("cross train")
                val_x, val_y = filter_and_fillna(val_x, val_y)
                model.train()
                # model_optim.zero_grad()
                outputs = model(val_x)
                pred_lgb = lgb_model.predict(val_x[:,-1,:].cpu().detach().numpy())
                cval_loss = IC_loss(torch.tensor(pred_lgb, requires_grad=True).to(device), outputs.requires_grad_(True), val_y.requires_grad_(True))
                cval_loss.backward()
                model_optim.step()
                cval_loss = cval_loss.item()
                # print(outputs.devie, outputs.requires_grad())
                # cval_ic = ic_between_timestep(outputs, val_y)
                cval_ic = pd.Series(outputs.cpu().detach().numpy()).corr(pd.Series(val_y.cpu().detach().numpy()))
            else:
                val_x, val_y = filter_and_fillna(val_x, val_y)
                model.eval()
                # with torch.no_grad():
                with torch.inference_mode():
                    outputs = model(val_x)
                    cval_loss = criterion(outputs, val_y)
                    cval_loss = cval_loss.item()
                    cval_ic = ic_between_timestep(outputs, val_y)

            if cval_loss is not None:
                val_loss.append(cval_loss)
            if cval_ic is not None:
                val_ic.append(cval_ic)

            if (time_step + 1) % 100 == 0:
                logger.info("\ttime_step: {0}, epoch: {1} | train_loss: {2:.7f} | cval_loss: {3:.7f} | cval_ic: {4:.4f}".format(time_step + 1, epoch + 1, loss.item(), cval_loss, cval_ic))
                speed = (time.time() - time_now) / (time_step + 1)
                left_time = speed * ((train_epochs - epoch) * train_steps - time_step)
                # logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

            if lradj == 'TST':
                scheduler.step()

        if train_loss:
            epoch_train_loss = np.mean(train_loss)
        else:
            raise ValueError("No training loss")
        if val_loss:
            epoch_val_loss = sum(val_loss) / len(val_loss)
        else:
            raise ValueError("No validation loss")
        epoch_val_ic = np.mean(val_ic) if val_ic else 0

        vali_loss, vali_ic_mean = validate_one_epoch(val_dataloader, model, criterion)
        test_loss, test_ic_mean = validate_one_epoch(test_dataloader, model, criterion)

        logger.info("Epoch: {} | Time: {:.2f}s".format(epoch + 1, time.time() - epoch_time))
        logger.info("Train Loss: {:.4f} | Val Loss: {:.4f} | Test Loss: {:.4f} | Val IC Mean: {:.4f}| Test IC Mean: {:.4f}".format(epoch_train_loss, vali_loss, test_loss, vali_ic_mean, test_ic_mean))
        early_model_path = os.path.join(exp_path, 'early_model.pth')
        early_stopping(vali_loss, model, early_model_path)
        if early_stopping.early_stop:
            logger.info("-----------Early stopping----------")
            break


        if lradj == 'TST':
            logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        elif lradj == 'cos':
            scheduler.step(epoch)
            logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        metrics.add_metrics(
            epoch=epoch,
            train_loss=epoch_train_loss,
            val_loss=vali_loss,
            test_loss=test_loss,
            val_ic_mean=vali_ic_mean,
            test_ic_mean=test_ic_mean,
            lr=model_optim.param_groups[0]['lr']
        )

        best_model_path = os.path.join(exp_path, 'best_model.pth')
        if best_ic < test_ic_mean:
            best_ic = test_ic_mean
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"best model saved with test ic : {best_ic:.4f}")

    last_model_path = os.path.join(exp_path, 'last_model.pth')
    torch.save(model.state_dict(), last_model_path)
    logger.info(f"last model saved with test ic : {test_ic_mean:.4f}")
    metrics.plot_loss(prefix="final_")
    metrics.plot_ic(prefix="final_")
    metrics.plot_lr(prefix="final_")
    df_metrics = metrics.to_dataframe()
    return model, df_metrics
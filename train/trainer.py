"""
Model Trainer Module - Encapsulates model training logic
"""

import os
import time
import random
from collections import deque
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from metrics.calculate_ic import ic_between_timestep
from metrics.train_plot import TrainingMetrics
from model.losses import IC_loss_double_diff, get_criterion
from utils.fillna import filter_and_fillna
from utils.tools import EarlyStopping


class ModelTrainer:
    """Model Trainer Class"""
    
    def __init__(self, config: Any, model: torch.nn.Module, device: str = 'cuda'):
        self.config = config
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self._init_components()
        self.best_ic = 0
        self.current_epoch = 0
        self.metrics = TrainingMetrics(config.exp_path)
        
    def _init_components(self):
        """Initialize training components"""
        self.criterion = get_criterion(self.config.loss).to(self.device)
        if self.config.cross_train:
            self.IC_loss = IC_loss_double_diff().to(self.device)
        
        # Optimizer
        if self.config.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping mechanism
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stop_patience, 
            verbose=True
        )
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler(device="cuda")
        
    def _create_scheduler(self) -> lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        if self.config.lradj == 'cos':
            warmup_epochs = 0
            decay_epochs = int(self.config.train_epochs * 5 / 5)
            return lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: (
                    (epoch / warmup_epochs) if epoch < warmup_epochs else
                    (0.5 * (1 + torch.cos(torch.tensor(
                        (epoch - warmup_epochs) / decay_epochs * torch.pi))))
                )
            )
        else:
            train_steps = 1000
            return lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                steps_per_epoch=train_steps,
                pct_start=self.config.pct_start,
                epochs=self.config.train_epochs,
                max_lr=self.config.learning_rate
            )
    
    def train_epoch(self, train_loader, logger) -> Tuple[float, torch.nn.Module]:
        """Train one epoch"""
        self.model.train()
        train_loss = []
        
        for time_step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            batch_x, batch_y_clean = filter_and_fillna(batch_x, batch_y)
            if batch_x.shape[0] == 0:
                continue
            
            self.optimizer.zero_grad()
            
            n_chunks = 4
            x_chunks = torch.chunk(batch_x, n_chunks, dim=0)
            y_chunks = torch.chunk(batch_y_clean, n_chunks, dim=0)
            
            for x_chunk, y_chunk in zip(x_chunks, y_chunks):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(x_chunk)
                    loss = self.criterion(outputs, y_chunk)
                    self.scaler.scale(loss / n_chunks).backward()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss.append(loss.item())
            
            if (time_step + 1) % 100 == 0:
                logger.info(
                    f"Time step: {time_step + 1}, Epoch: {self.current_epoch + 1} | "
                    f"Training loss: {loss.item():.7f}"
                )
            
            if self.config.lradj == 'TST':
                self.scheduler.step()
        
        if not train_loss:
            raise ValueError("No training loss data")
        
        return np.mean(train_loss), self.model
    
    def validate(self, val_loader, n_chunks: int = 4) -> Tuple[float, float]:
        """Validate model performance"""
        self.model.eval()
        total_loss = []
        ic_list = []
        
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.float().to(self.device), y.float().to(self.device)
                x, y = filter_and_fillna(x, y)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                ic = ic_between_timestep(outputs, y)
                
                total_loss.append(loss.item())
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        avg_loss = np.mean(total_loss)
        avg_ic = np.mean(ic_list) if ic_list else 0
        
        return avg_loss, avg_ic
    
    def train(self, train_loader, val_loader, test_loader, logger) -> Tuple[torch.nn.Module, pd.DataFrame]:
        """Complete training pipeline"""
        os.makedirs(os.path.join(self.config.exp_path, 'models/'), exist_ok=True)
        
        logger.info(f"Starting training, device: {self.device}")
        
        for epoch in range(self.config.train_epochs):
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            train_loss, self.model = self.train_epoch(train_loader, logger)
            val_loss, val_ic = self.validate(val_loader)
            test_loss, test_ic = self.validate(test_loader)
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch: {epoch + 1} | Time: {epoch_time:.2f}s")
            logger.info(
                f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Test loss: {test_loss:.4f} | Val IC: {val_ic:.4f} | Test IC: {test_ic:.4f}"
            )
            
            early_model_path = os.path.join(self.config.exp_path, 'models/early_model.pth')
            self.early_stopping(val_loss, self.model, early_model_path)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
            
            if self.config.lradj == 'TST':
                logger.info(f'Learning rate updated to: {self.scheduler.get_last_lr()[0]}')
            elif self.config.lradj == 'cos':
                self.scheduler.step(epoch)
                logger.info(f'Learning rate updated to: {self.scheduler.get_last_lr()[0]}')
            
            self.metrics.add_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                val_ic_mean=val_ic,
                test_ic_mean=test_ic,
                lr=self.optimizer.param_groups[0]['lr']
            )
            
            if self.best_ic < test_ic:
                self.best_ic = test_ic
                best_model_path = os.path.join(self.config.exp_path, 'models/best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f"Saved best model, test IC: {self.best_ic:.4f}")
        
        last_model_path = os.path.join(self.config.exp_path, 'models/last_model.pth')
        torch.save(self.model.state_dict(), last_model_path)
        logger.info(f"Saved final model, test IC: {test_ic:.4f}")
        
        self.metrics.plot_loss(prefix="final_")
        self.metrics.plot_ic(prefix="final_")
        self.metrics.plot_lr(prefix="final_")
        
        return self.model, self.metrics.to_dataframe()
    
    def test(self, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        """Model testing"""
        self.model.eval()
        pred_list = []
        true_list = []
        
        with torch.inference_mode():
            for x, y in tqdm(test_loader, desc="Testing"):
                x, y = x.float().to(self.device), y.float().to(self.device)
                x = torch.nan_to_num(x, nan=0)
                
                pred = self.model(x)
                pred_cpu = pred.squeeze().detach().cpu().numpy()
                y_cpu = y.squeeze().detach().cpu().numpy()
                
                true_list.append(y_cpu)
                pred_list.append(pred_cpu)
        
        true_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in true_list], axis=0)
        pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)
        
        return pred_arr, true_arr


def create_trainer(config: Any, model: torch.nn.Module, device: str = 'cuda') -> ModelTrainer:
    """Factory function to create trainer instance"""
    return ModelTrainer(config, model, device)


# ==================== Backward Compatibility Functions ====================

def norm_train(config, train_dataloader, val_dataloader, test_dataloader, model, early_stopping, model_optim, criterion, scheduler, logger, scaler):
    """
    Standard training function - backward compatibility
    Uses new trainer implementation but maintains original interface
    """
    # Create trainer
    trainer = ModelTrainer(config, model, config.device)
    
    # Replace trainer components to match passed parameters
    trainer.optimizer = model_optim
    trainer.criterion = criterion
    trainer.scheduler = scheduler
    trainer.early_stopping = early_stopping
    trainer.scaler = scaler
    
    # Execute training
    trained_model, df_metrics = trainer.train(train_dataloader, val_dataloader, test_dataloader, logger)
    
    return trained_model, df_metrics


def train_and_cross_time_train(config, train_dataloader, val_dataloader, test_dataloader, model, early_stopping, model_optim, criterion, scheduler, logger, scaler):
    """
    Cross-time training function - backward compatibility
    Uses new trainer implementation but maintains original interface
    """
    # Create trainer
    trainer = ModelTrainer(config, model, config.device)
    
    # Replace trainer components to match passed parameters
    trainer.optimizer = model_optim
    trainer.criterion = criterion
    trainer.scheduler = scheduler
    trainer.early_stopping = early_stopping
    trainer.scaler = scaler
    
    # Execute training
    trained_model, df_metrics = trainer.train(train_dataloader, val_dataloader, test_dataloader, logger)
    
    return trained_model, df_metrics


def GRU_fin_test(test_loader, model):
    """
    GRU final test function - backward compatibility
    """
    device = next(model.parameters()).device
    model.eval()
    pred_list = []
    true_list = []
    
    with torch.inference_mode():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.float().to(device), y.float().to(device)
            x = torch.nan_to_num(x, nan=0)
            
            pred = model(x)
            pred_cpu = pred.squeeze().detach().cpu().numpy()
            y_cpu = y.squeeze().detach().cpu().numpy()
            
            true_list.append(y_cpu)
            pred_list.append(pred_cpu)
    
    true_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in true_list], axis=0)
    pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)
    
    return pred_arr, true_arr


def GRU_fin_test_new(test_loader, model, n_chunks=4):
    """
    Supports chunked inference to reduce memory usage
    """
    device = next(model.parameters()).device
    model.eval()
    pred_list = []
    true_list = []
    
    with torch.inference_mode():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.float().to(device), y.float().to(device)
            x = torch.nan_to_num(x, nan=0)
            
            # Chunked inference
            if x.shape[0] > n_chunks:
                x_chunks = torch.chunk(x, n_chunks, dim=0)
                y_chunks = torch.chunk(y, n_chunks, dim=0)
                outputs_list = []
                
                for x_chunk, y_chunk in zip(x_chunks, y_chunks):
                    output = model(x_chunk)
                    outputs_list.append(output)
                
                pred = torch.cat(outputs_list, dim=0)
            else:
                pred = model(x)
            
            pred_cpu = pred.squeeze().detach().cpu().numpy()
            y_cpu = y.squeeze().detach().cpu().numpy()
            
            true_list.append(y_cpu)
            pred_list.append(pred_cpu)
    
    true_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in true_list], axis=0)
    pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)
    
    return pred_arr, true_arr


def GRU_pred_market_new(mkt_align_x_loader, model, n_chunks=4):
    """
    GRU market prediction function (new version) - backward compatibility
    """
    device = next(model.parameters()).device
    model.eval()
    pred_list = []
    
    with torch.inference_mode():
        for x in tqdm(mkt_align_x_loader, desc="Predicting"):
            x = x.float().to(device)
            x = torch.nan_to_num(x, nan=0)
            
            # Chunked inference
            if x.shape[0] > n_chunks:
                x_chunks = torch.chunk(x, n_chunks, dim=0)
                outputs_list = []
                
                for x_chunk in x_chunks:
                    output = model(x_chunk)
                    outputs_list.append(output)
                
                pred = torch.cat(outputs_list, dim=0)
            else:
                pred = model(x)
            
            pred_cpu = pred.squeeze().detach().cpu().numpy()
            pred_list.append(pred_cpu)
    
    pred_arr = np.concatenate([p.reshape(-1, p.shape[0]) for p in pred_list], axis=0)
    return pred_arr
import torch
from typing import *
import numpy as np
from tqdm import tqdm
import os

from torch.utils.data import DataLoader

from .metrics import Metric
from .callbacks import Callback
from .handlers import MetricsHandler, CallbacksHandler


class CustomModel:


    def __init__(self,
                 model: torch.nn.Module,
                 loss: Callable, 
                 optimizer: torch.optim.Optimizer, 
                 metrics: List[Metric] = [],
                ) -> None:
        
        def validate():
            """ Validate model, optimizer"""
            if not isinstance(self.model, torch.nn.Module):
                raise TypeError("model argument must be a torch.nn.Module")
            
            if not isinstance(self.optimizer, torch.optim.Optimizer):
                raise TypeError("optimizer argument must be a torch.optim.Optimizer")
            
        self.model = model
        self.optimizer = optimizer
        self.loss = loss


        self.metrics = MetricsHandler(metrics)

        self.stop_training = False
        self._device = next(self.model.parameters()).device

        validate()


    def toDevice(self, 
                 data: np.ndarray, 
                 target: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        if data.device != self._device:
            data = data.to(self._device)

        if target.device != self._device:
            target = target.to(self._device)

        return data, target
    
    def training_step(self,
                      data: Union[np.ndarray, torch.Tensor], 
                      target: Union[np.ndarray, torch.Tensor]
                     ) -> torch.Tensor:
    

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss(output, target)

        loss.backward()
        self.optimizer.step()
        torch.cuda.synchronize() 

        self.metrics.update(target, output)

        return loss
    

    def tesing_step(self,
                    data: Union[np.ndarray, torch.Tensor], 
                    target: Union[np.ndarray, torch.Tensor]
                    ) -> torch.Tensor:
        
        output = self.model(data)
        loss = self.loss(output, target)

        self.metrics.update(target, output)

        return loss
    

    def save(self, 
             path: str):

        torch.save(self.model.state_dict(), path)


    def train(self, 
              dataProvider: DataLoader):
        
        self.model.train()

        loss_sum = 0
        pbar = tqdm(dataProvider, total=len(dataProvider))
        for step, (data, target) in enumerate(pbar, start=1):
            self.callbacks.on_batch_begin(step, logs=None, train=True)

            data, target = self.toDevice(data, target)
            loss = self.train_step(data, target)
            loss_sum += loss.item()
            loss_mean = loss_sum / step


            logs = self.metrics.results(loss_mean, train=True)
            description = self.metrics.description(epoch=self._epoch, train=True)


            pbar.set_description(description)

            self.callbacks.on_batch_end(step, logs=logs, train=True)

 
        self.metrics.reset()


        dataProvider.on_epoch_end()

        return logs



    def test(self, 
             dataProvider: DataLoader):
        
        self.model.eval()
        loss_sum = 0
        pbar = tqdm(dataProvider, total=len(dataProvider))
        for step, (data, target) in enumerate(pbar, start=1):
            self.callbacks.on_batch_begin(step, logs=None, train=False)

            data, target = self.toDevice(data, target)
            loss = self.test_step(data, target)
            loss_sum += loss.item()
            loss_mean = loss_sum / step

            logs = self.metrics.results(loss_mean, train=False)
            description = self.metrics.description(train=False)


            pbar.set_description(description)

            self.callbacks.on_batch_end(step, logs=logs, train=False)


        self.metrics.reset()


        dataProvider.on_epoch_end()

        return logs
    

    def fit(
        self, 
        train_dataProvider: DataLoader, 
        test_dataProvider: DataLoader, 
        epochs: int, 
        initial_epoch:int = 1, 
        callbacks: List[Callback] = [],
        save_every:int=5,
        save_path:str="./"
        ) -> dict:
        
        self.save_path = os.path.join(save_path, "checkpoint")
        os.makedirs(self.save_path, exist_ok=True)

        self._epoch = initial_epoch
        self.callbacks = CallbacksHandler(self, callbacks)
        self.callbacks.on_train_begin()
        for epoch in range(initial_epoch, initial_epoch + epochs):
            self.callbacks.on_epoch_begin(epoch)

            train_logs = self.train(train_dataProvider)
            val_logs = self.test(test_dataProvider)

            logs = {**train_logs, **val_logs}
            self.callbacks.on_epoch_end(epoch, logs=logs)

            if self.stop_training:
                break
            
            if epoch%save_every == 0:
                self.save(path=self.save_path)
            self._epoch += 1



        self.callbacks.on_train_end(logs)

        return logs
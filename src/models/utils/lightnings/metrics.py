import torch
from typing import *
import numpy as np
from itertools import groupby

from .text_utils import get_cer, get_wer


class Metric:

    def __init__(self, 
                 name: str) -> None:

        self.name = name

    def reset(self):

        self.__init__()

    def update(self, 
               output: torch.Tensor, 
               target: torch.Tensor):

        pass

    def result(self):

        pass


class Accuracy(Metric):

    def __init__(self, 
                 name='accuracy') -> None:
        
        super().__init__(name=name)
        self.correct = 0
        self.total = 0

    def update(self, 
               output: torch.Tensor, 
               target: torch.Tensor):

        _, predicted = torch.max(output.data, 1)
        self.total += target.size(0)
        self.correct += (predicted == target).sum().item()

    def result(self):

        return self.correct / self.total


class CERMetric(Metric):

    def __init__(
        self, 
        vocabulary: Union[str, list],
        name: str='CER'
    ) -> None:
        
        super().__init__(name=name)
        self.vocabulary = vocabulary
        self.reset()

    def reset(self):
 
        self.cer = 0
        self.counter = 0

    def update(self, 
               output: torch.Tensor, 
               target: torch.Tensor) -> None:
        
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        argmax_preds = np.argmax(output, axis=-1)
        
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

        output_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        target_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in target]

        cer = get_cer(output_texts, target_texts)

        self.cer += cer
        self.counter += 1

    def result(self) -> float:
        return self.cer / self.counter
    

class WERMetric(Metric):

    def __init__(
        self, 
        vocabulary: Union[str, list],
        name: str='WER'
    ) -> None:
        
        super().__init__(name=name)
        self.vocabulary = vocabulary
        self.reset()

    def reset(self):

        self.wer = 0
        self.counter = 0

    def update(self, 
               output: torch.Tensor, 
               target: torch.Tensor) -> None:


        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        argmax_preds = np.argmax(output, axis=-1)
        

        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]


        output_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        target_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in target]

        wer = get_wer(output_texts, target_texts)

        self.wer += wer
        self.counter += 1

    def result(self) -> float:
        
        return self.wer / self.counter
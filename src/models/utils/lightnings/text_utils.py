
from typing import *
import numpy as np
from itertools import groupby
    
def ctc_decoder(predictions: np.ndarray, chars: Union[str, list]) -> List[str]:

    argmax_preds = np.argmax(predictions, axis=-1)
    

    grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]


    texts = ["".join([chars[k] for k in group if k < len(chars)]) for group in grouped_preds]

    return texts


def edit_distance(prediction_tokens: List[str], 
                  reference_tokens: List[str]) -> int:

    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]


    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    
    dp[0] = [j for j in range(len(reference_tokens) + 1)]


    for i, p_tok in enumerate(prediction_tokens):
        for j, r_tok in enumerate(reference_tokens):

            if p_tok == r_tok:
                dp[i+1][j+1] = dp[i][j]

            else:
                dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1


    return dp[-1][-1]

def get_cer(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
    ) -> float:

    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]

    total, errors = 0, 0
    for pred_tokens, tgt_tokens in zip(preds, target):
        errors += edit_distance(list(pred_tokens), list(tgt_tokens))
        total += len(tgt_tokens)

    if total == 0:
        return 0.0

    cer = errors / total

    return cer

def get_wer(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
    ) -> float:

    if isinstance(preds, str):
        preds = preds.split()
    if isinstance(target, str):
        target = target.split()

    errors = edit_distance(preds, target)
    total_words = len(target)

    if total_words == 0:
        return 0.0

    return errors / total_words
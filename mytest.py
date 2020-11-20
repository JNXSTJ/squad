from json import dumps
from tqdm import tqdm
import time
import util
import torch
import layers
import torch.nn as nn
from ujson import load as json_load

def test1():
    batch = 1
    c_len = 6

    start = torch.randn(batch, c_len)
    end = torch.randn(batch, c_len)
    start = torch.softmax(start, -1)
    end = torch.softmax(end, -1)
    start_ids, end_ids = util.discretize(start, end, max_len = 3,  no_answer = True)
    return start_ids, end_ids

def test2():
    batch = 4
    c_len = 10
    q_len = 6
    hidden_size = 8
    c = torch.randn(batch, c_len, hidden_size * 2)
    q = torch.randn(batch, q_len, hidden_size * 2)
    BiDAFA = layers.BiDAFAttention(hidden_size * 2)
    ret = BiDAFA.forward(c, q, None, None)
    return ret

if __name__ == '__main__':
    test2()

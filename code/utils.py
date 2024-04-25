import parse
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
from parse import args


def getlabel(test_data, pred_data):
    r, recall_n = [], []
    for i in range(len(pred_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        if len(groundTrue) > 0:
            r.append(list(map(lambda x: x in groundTrue, predictTopK)))
            recall_n.append(len(groundTrue))
    return np.array(r), recall_n


def test(sorted_items, groundTrue):
    sorted_items = sorted_items.cpu().numpy()
    r, recall_n = getlabel(groundTrue, sorted_items)
    pre, recall, ndcg, ndcg2 = [], [], [], []
    for k in parse.topks:
        now_k = min(k, r.shape[1])
        pred = r[:, :now_k]
        right_pred = pred.sum(1)
        # precision
        pre.append(np.sum(right_pred / now_k))
        # recall
        recall.append(np.sum(right_pred/recall_n))
        # ndcg
        dcg = np.sum(pred * (1. / np.log2(np.arange(2, now_k + 2))), axis=1)
        d_val = [np.sum(1. / np.log2(np.arange(2, i + 2)))
                 for i in range(0, now_k + 1)]
        idcg = np.array([d_val[int(i)] for i in np.minimum(recall_n, now_k)])
        ndcg.append(np.sum(dcg / idcg))
    return torch.tensor(pre), torch.tensor(recall), torch.tensor(ndcg)


def sum_norm(indices, values, n):
    s = torch.zeros(n, device=values.device).scatter_add(0, indices[0], values)
    s[s == 0.] = 1.
    return values/s[indices[0]]


def sparse_softmax(indices, values, n):
    return sum_norm(indices, torch.clamp(torch.exp(values), min=-5, max=5), n)

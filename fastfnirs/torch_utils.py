import numpy as np
import torch


def sub_dict_to_tensor(X, y):
    subject_ix = np.array([s for s in X.keys() for _ in range(len(X[s]))])
    X = torch.tensor(np.concatenate(list(X.values()))).float()
    y = torch.tensor(np.concatenate(list(y.values()))).long()
    return X, y, subject_ix

import numpy as np
import torch


def create_dataset(X, y):
    X_tensor = (
        X if isinstance(X, torch.Tensor) else torch.tensor(X, dtype=torch.float32)
    )
    y_tensor = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.long)
    return torch.utils.data.TensorDataset(X_tensor, y_tensor)


def sub_dict_to_tensor(X, y):
    subject_ix = np.array([s for s in X.keys() for _ in range(len(X[s]))])
    X = torch.tensor(np.concatenate(list(X.values()))).float()
    y = torch.tensor(np.concatenate(list(y.values()))).long()
    return X, y, subject_ix

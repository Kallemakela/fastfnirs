import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from SKUtils.StandardScalerND import StandardScalerND
from fastfnirs.classification.classification import get_cv_splits_from_arg
from fastfnirs.deep_learning.utils import ValScoring
from fastfnirs.classification import get_cv_from_str


def ma_score(net, X=None, y=None, n_epochs=5, score_name="valid_acc"):
    """Moving average score over the last n_epochs"""
    accs = net.history[-n_epochs:, score_name]
    return np.mean(accs)


def get_model(
    base_model,
    max_epochs,
    net_params,
    train_split=None,
    criterion=nn.CrossEntropyLoss,
    **kwargs,
):
    skorch_net = NeuralNetClassifier(
        base_model,
        max_epochs=max_epochs,
        # train_split=ValidSplit(10, stratified=True),  # skorch only uses the first split so train_split can't be directly used
        train_split=train_split,
        criterion=criterion,
        verbose=0,
        **net_params,
        **kwargs,
    )

    # to remove callback
    # skorch_net.set_params(callbacks__valid_acc=None)

    pipeline = Pipeline([("scaler", StandardScalerND()), ("net", skorch_net)])
    skorch_net.pipeline = pipeline  # to access pipeline during scoring callback
    return pipeline


class NestedCVSkorch:
    def __init__(
        self,
        base_model,
        outer_cv,
        inner_cv=None,
        net_params={},
        default_callbacks=[],
        max_epochs=None,
        scoring=accuracy_score,
        get_model_kwargs={},
        tqdm_kwargs={},
    ):
        self.net_params = net_params
        self.default_callbacks = default_callbacks
        self.base_model = base_model
        self.max_epochs = max_epochs
        self.scoring = scoring
        self.get_model_kwargs = get_model_kwargs
        self.tqdm_kwargs = tqdm_kwargs
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.outer_split_info = []

    def fit(self, X, y):
        outer_splits = get_cv_splits_from_arg(self.outer_cv, X=X, y=y)
        for i, (o_tr_ix, o_te_ix) in tqdm(enumerate(outer_splits), **self.tqdm_kwargs):
            X_tr, y_tr = X[o_tr_ix], y[o_tr_ix]

            # inner CV
            best_val_epoch = None
            best_val_score = -np.inf
            inner_scores = []
            if self.inner_cv is not None:
                for ii, (i_tr_ix, i_te_ix) in enumerate(
                    self.inner_cv.split(X_tr, y_tr)
                ):
                    X_tr_i, y_tr_i = X_tr[i_tr_ix], y_tr[i_tr_ix]
                    X_val, y_val = X_tr[i_te_ix], y_tr[i_te_ix]
                    callbacks = self.default_callbacks + [
                        ValScoring(X_val, y_val, scorer=self.scoring, name="val_acc")
                    ]
                    inner_split_model = get_model(
                        self.base_model,
                        self.max_epochs,
                        self.net_params,
                        callbacks=callbacks,
                        **self.get_model_kwargs,
                    )
                    inner_split_model.fit(X_tr_i, y_tr_i)
                    isplit_val_accs = np.array(
                        [
                            h["val_acc"]
                            for h in inner_split_model.named_steps["net"].history
                        ]
                    )
                    inner_scores.append(isplit_val_accs)

                # Add logic here to choose the best param set
                # for now just takes the best epoch
                inner_scores_mean = np.array(inner_scores).mean(axis=0)
                best_val_epoch = np.argmax(inner_scores_mean)
                best_val_score = inner_scores_mean[best_val_epoch]

            # refit and test
            X_te, y_te = X[o_te_ix], y[o_te_ix]
            callbacks = self.default_callbacks + [
                ValScoring(X_te, y_te, scorer=self.scoring, name="test_acc")
            ]
            test_model = get_model(
                self.base_model,
                self.max_epochs,
                self.net_params,
                callbacks=callbacks,
                **self.get_model_kwargs,
            )
            test_model.fit(X_tr, y_tr)
            test_scores = np.array(
                [h["test_acc"] for h in test_model.named_steps["net"].history]
            )
            self.outer_split_info.append(
                {
                    "outer_split": o_tr_ix,
                    "test_scores": test_scores,
                    "inner_scores": inner_scores,
                    "best_val_epoch": best_val_epoch,
                    "best_val_score": best_val_score,
                }
            )
        return self

    @property
    def test_scores(self):
        return np.array([o["test_scores"] for o in self.outer_split_info])

    @property
    def inner_scores(self):
        return np.array([o["inner_scores"] for o in self.outer_split_info])

    @property
    def best_val_epochs(self):
        return np.array([o["best_val_epoch"] for o in self.outer_split_info])

    @property
    def best_val_scores(self):
        return np.array([o["best_val_score"] for o in self.outer_split_info])

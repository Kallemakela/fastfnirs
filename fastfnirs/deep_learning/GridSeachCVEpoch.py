from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback, LRScheduler
from itertools import product
import torch.nn as nn
import time

from SKUtils.StandardScalerND import StandardScalerND
from fastfnirs.utils import save_to
from fastfnirs.utils.basic import centered_moving_average
from fastfnirs.classification import get_cv_from_str


class GridSeachCVEpoch:
    def __init__(self, net, param_grid, scoring=None, save_path=None):
        self.net = net
        self.param_grid = param_grid
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = 0
        self.best_epoch_ = 0
        self.best_estimator_ = None
        self.info = {}
        self.save_path = save_path
        self.norm_dim = None
        self.overwrite = False

        self.lrs_step_size = 50
        self.lrs_gamma = 0.5

    @property
    def params(self):
        return [p["params"] for p in self.info.values()]

    def get_lr_scheduler(self):
        return LRScheduler(
            policy="StepLR", step_size=self.lrs_step_size, gamma=self.lrs_gamma
        )

    def get_val_splits(self, X, y, cv):
        if isinstance(cv, str):
            self.cv = get_cv(cv)
            return list(self.cv.split(X, y))
        elif isinstance(cv, list):
            self.cv = cv
            return cv
        else:
            self.cv = cv
            return list(self.cv.split(X, y))

    def fit(self, X, y, cv="k10", device="cpu"):
        val_splits = self.get_val_splits(X, y, cv)
        param_combs = sorted(product(*self.param_grid.values()))
        for pi, params in enumerate(param_combs):
            try:
                t0 = time.time()
                params_dict = dict(zip(self.param_grid.keys(), params))
                train_scores, val_scores = [], []
                print(f"Param set {pi}: {params_dict}")
                if self.info.get(pi) is not None and not self.overwrite:
                    print("Already trained, skipping")
                    continue
                for inner_split_ix, (train_index, val_index) in enumerate(val_splits):
                    print(
                        f"{inner_split_ix:2} n_train_sub={len(np.unique(self.train_subject_ids[train_index]))}, val_subs={np.unique(self.train_subject_ids[val_index])}"
                    )
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]
                    train_scoring = DatasetScoring(X_train, y_train, accuracy_score)
                    validation_scoring = DatasetScoring(X_val, y_val, accuracy_score)
                    estimator = self._get_estimator(
                        params_dict,
                        extra_callbacks=[
                            validation_scoring,
                            train_scoring,
                        ],
                        device=device,
                    )
                    estimator.fit(X_train, y_train)
                    val_scores.append(validation_scoring.scores_)
                    train_scores.append(train_scoring.scores_)

                avg_scores = np.mean(val_scores, axis=0)
                max_score = np.max(avg_scores)
                max_score_epoch = np.argmax(avg_scores) + 1
                if max_score > self.best_score_:
                    self.best_score_ = max_score
                    self.best_params_ = params_dict
                    self.best_epoch_ = max_score_epoch
                    self.best_estimator_ = estimator

                params_info = {
                    "params": params_dict,
                    "val_scores": val_scores,
                    "train_scores": train_scores,
                }
                self.info[pi] = params_info

                if self.save_path is not None:
                    save_to(self, self.save_path)
                    print(f"Saved to {self.save_path}")

                print(f"Param set {pi} done in {(time.time()-t0) / 60:.1f} mins")

            except Exception as e:
                print(f"\nEncountered error with params {params_dict}:\n\n{e}\n")
                continue

        return self

    def _get_estimator(self, params, extra_callbacks=[], device="cpu"):
        """Get a new estimator with the given params and callbacks"""
        callbacks = [self.get_lr_scheduler()]
        callbacks.extend(extra_callbacks)
        clf = NeuralNetClassifier(
            self.net,
            train_split=None,
            criterion=nn.CrossEntropyLoss,
            verbose=0,
            device=device,
            callbacks=callbacks,
            **params,
        )
        estimator = Pipeline(
            [
                ("scaler", StandardScalerND(dim=self.norm_dim)),
                ("clf", clf),
            ]
        )
        clf.pipeline = estimator  # to enable access to pipeline in scoring
        return estimator

    def set_device(self, device):
        self.best_estimator_.named_steps["clf"].device = device
        return self

    def refit(self, X, y, X_test=None, y_test=None, max_epochs=None):
        params = self.best_params_.copy()
        params["max_epochs"] = max_epochs or self.best_epoch_
        test = X_test is not None and y_test is not None
        extra_callbacks = []
        if test:
            test_scoring = DatasetScoring(X_test, y_test, accuracy_score)
            extra_callbacks.append(test_scoring)
        self.best_estimator_ = self._get_estimator(
            params, extra_callbacks=extra_callbacks
        )
        self.best_estimator_.fit(X, y)
        if test:
            self.test_scores_ = test_scoring.scores_
        return self

    def refit_all_params(self, X, y, X_test=None, y_test=None, max_epochs=None):
        """Refit the model with each parameter combination and store results.

        Args:
            X: Training data
            y: Training labels
            X_test: Optional test data
            y_test: Optional test labels
            max_epochs: Maximum number of epochs to train for each parameter combination

        Returns:
            self: Returns the instance with updated results
        """

        test = X_test is not None and y_test is not None
        self.refit_results_ = {}
        for pi, pinfo in self.info.items():
            params_dict = pinfo["params"]

            if max_epochs is None:
                max_epochs = self.best_epoch_
            elif callable(max_epochs):
                best_epoch, best_score = self._find_best_epoch_for_params(
                    pi, max_epochs
                )
                max_epochs = best_epoch + 1

            params_dict["max_epochs"] = max_epochs

            extra_callbacks = []
            if test:
                test_scoring = DatasetScoring(X_test, y_test, accuracy_score)
                extra_callbacks.append(test_scoring)

            estimator = self._get_estimator(
                params_dict, extra_callbacks=extra_callbacks
            )
            estimator.fit(X, y)

            result = {"params": params_dict, "max_epochs": max_epochs}

            if test:
                result["test_scores"] = test_scoring.scores_

            self.refit_results_[pi] = result

        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)

    def get_search_epoch_score(self):
        search_epoch_test_score = self.test_scores_[self.best_epoch_ - 1]
        return self.best_epoch_, search_epoch_test_score

    def _find_best_epoch_for_params(self, pi, score_processor):
        """Find the best epoch and score for a specific parameter combination.

        Args:
            pi: Parameter combination index
            score_processor (callable): Function that takes average scores and returns processed scores

        Returns:
            tuple: (best_epoch, best_score) for the given parameter combination
        """
        val_scores = self.info[pi]["val_scores"]
        avg_scores = np.mean(val_scores, axis=0)
        processed_scores = score_processor(avg_scores)
        best_score = np.max(processed_scores)
        best_epoch = np.argmax(processed_scores)
        return best_epoch, best_score

    def _find_best_epoch_params(self, score_processor):
        """Helper method to find best epoch and parameters using a custom score processing function.

        Args:
            score_processor (callable): Function that takes average scores and returns processed scores

        Returns:
            tuple: (best_epoch, best_score, best_params)
        """
        best_score = -np.inf
        best_epoch = None
        best_params = None

        for pi in self.info:
            epoch, score = self._find_best_epoch_for_params(pi, score_processor)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_params = self.info[pi]["params"]

        return best_epoch, best_score, best_params

    def get_clipped_best(self, start_epoch=5):
        """Find the best parameter combination and epoch based on validation scores starting from given epoch.

        Args:
            start_epoch (int): First epoch to consider in the search.

        Returns:
            tuple: (best_epoch, best_score, best_params) found from start_epoch onwards
        """
        return self._find_best_epoch_params(
            lambda scores: clip_scores(scores, start_epoch)
        )

    def get_moving_average_best(self, window_size=11):
        """Find the best parameter combination and epoch based on validation scores using centered moving average.

        Args:
            window_size (int): Size of the window for centered moving average.

        Returns:
            tuple: (best_epoch, best_score, best_params) found using moving average
        """
        return self._find_best_epoch_params(
            lambda scores: moving_average_scores(scores, window_size)
        )

    def get_clipped_moving_average_best(self, start_epoch=5, window_size=11):
        """Find the best parameter combination and epoch based on validation scores using both clipping and moving average.

        Args:
            start_epoch (int): First epoch to consider in the search.
            window_size (int): Size of the window for centered moving average.

        Returns:
            tuple: (best_epoch, best_score, best_params) found using both clipping and moving average
        """
        return self._find_best_epoch_params(
            lambda scores: clip_moving_average_scores(scores, start_epoch, window_size)
        )

    def update_best(self, epoch, score, params):
        self.best_epoch_ = epoch
        self.best_score_ = score
        self.best_params_ = params

    def get_clipped_search_epoch_score(self, start_epoch=5):
        """Get the test score at the best epoch found from start_epoch onwards.

        Args:
            start_epoch (int): First epoch to consider in the search.

        Returns:
            tuple: (best_epoch, test_score_at_best_epoch)
        """
        best_epoch, best_score, best_params = self.get_clipped_best(start_epoch)
        print(best_epoch, best_score, best_params)
        print(self.best_epoch_, self.best_score_, self.best_params_)
        search_epoch_test_score = self.test_scores_[best_epoch - 1]
        return best_epoch, search_epoch_test_score

    def get_last_epoch_score(self):
        last_epoch = len(self.test_scores_)
        last_epoch_test_score = self.test_scores_[-1]
        return last_epoch, last_epoch_test_score

    def get_max_epoch_score(self):
        max_epoch = np.argmax(self.test_scores_) + 1
        max_epoch_test_score = np.max(self.test_scores_)
        return max_epoch, max_epoch_test_score

    def plot_val_curves(self):
        for pi in self.info:
            params = self.info[pi]["params"]
            scores = np.array(self.info[pi]["val_scores"])
            scores_mean = np.mean(scores, axis=0)  # mean accross inner splits
            ma = centered_moving_average(scores_mean)
            if len(np.array(scores).shape) == 1:
                print([len(s) for s in scores])
            plt.plot(np.arange(1, len(ma) + 1), ma, label=f"P{pi}")
            plt.axvline(self.best_epoch_, color="black", linestyle="--")
            plt.xlabel("Epoch")
            plt.ylabel("Val Score")
        plt.legend()

    def plot_test_curve(self):
        last_epoch = len(self.test_scores_)
        plt.plot(np.arange(1, last_epoch + 1), self.test_scores_, label="test")
        plt.axvline(self.best_epoch_, color="r")


def clip_scores(scores, start_epoch=5):
    """Clip scores by setting all scores before start_epoch to minimum score.

    Args:
        scores (np.ndarray): Array of scores
        start_epoch (int): First epoch to consider, defaults to 5

    Returns:
        np.ndarray: Clipped scores
    """
    min_score = np.min(scores)
    clipped = scores.copy()
    clipped[:start_epoch] = min_score
    return clipped


def moving_average_scores(scores, window_size=11):
    """Apply centered moving average to scores.

    Args:
        scores (np.ndarray): Array of scores
        window_size (int): Size of the moving average window, defaults to 11

    Returns:
        np.ndarray: Smoothed scores
    """
    return centered_moving_average(scores, window_size=window_size)


def clip_moving_average_scores(scores, start_epoch=5, window_size=11):
    """Apply both clipping and moving average to scores.

    Args:
        scores (np.ndarray): Array of scores
        start_epoch (int): First epoch to consider, defaults to 5
        window_size (int): Size of the moving average window, defaults to 11

    Returns:
        np.ndarray: Processed scores
    """
    ma = moving_average_scores(scores, window_size)
    return clip_scores(ma, start_epoch)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from tqdm import tqdm

from fastfnirs.classification import get_cv_from_str


class NestedCV:
    def __init__(
        self,
        model,
        param_grid,
        outer_cv,
        inner_cv,
        scorer_func=accuracy_score,
        tqdm_kwargs={},
        gs_kwargs={},
    ):
        self.model = model
        self.param_grid = param_grid
        self.tqdm_kwargs = tqdm_kwargs
        self.gs_kwargs = gs_kwargs
        self.scorer_func = scorer_func
        self.outer_cv = (
            get_cv_from_str(outer_cv) if isinstance(outer_cv, str) else outer_cv
        )
        self.inner_cv = (
            get_cv_from_str(inner_cv) if isinstance(inner_cv, str) else inner_cv
        )

    def fit(self, X, y):
        outer_splits = list(self.outer_cv.split(X, y))
        self.split_info = []
        self.test_scores = []
        for o_tr_ix, o_test_ix in tqdm(outer_splits, **self.tqdm_kwargs):
            Xs_train, ys_train = X[o_tr_ix], y[o_tr_ix]
            Xs_test, ys_test = X[o_test_ix], y[o_test_ix]
            gs = GridSearchCV(
                clone(self.model),
                self.param_grid,
                cv=self.inner_cv,
                refit=True,
                **self.gs_kwargs,
            )
            gs.fit(Xs_train, ys_train)
            y_pred = gs.predict(Xs_test)
            self.test_scores.append(self.scorer_func(ys_test, y_pred))
            self.split_info.append(gs.cv_results_)
        return self

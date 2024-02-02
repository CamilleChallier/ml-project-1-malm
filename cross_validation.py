from typing import Union

import numpy as np

from base_regressors import Regressor


class CrossValidation:
    def __init__(
        self,
        model: Regressor,
        k_folds: int,
        metric,
        seed: int = 1,
    ) -> None:
        """Cross validation class

        Parameters
        ----------
        model : Regressor
            The model to be cross validated, must have a `fit` and `predict` method
        k_folds : int
            The number of folds to be used
        metric : callable
            A function that takes two arguments, the first being the predicted values and the second the true values
        seed : int, optional
            The seed to be used for the random number generator, by default 1
        """
        self.model = model
        self.k_folds = k_folds
        self.seed = seed
        self.metric = metric

    def set_seed(self) -> None:
        """Set the seed for the random number generator"""
        np.random.seed(self.seed)

    def build_k_indices(self, y: np.ndarray) -> np.ndarray:
        """Vectorized implementation of k-fold indices building

        Parameters
        ----------
        y : np.ndarray
            The labels, used to determine the number of rows

        Returns
        -------
        np.ndarray
            A 2D array of shape=(k_folds, N/k_folds) that indicates the data indices for each fold
        """
        num_row = y.shape[0]
        num_row = int(num_row / self.k_folds) * self.k_folds
        self.set_seed()
        return np.random.permutation(num_row).reshape(
            (self.k_folds, int(num_row / self.k_folds))
        )

    def _evaluate_fold(
        self, X: np.ndarray, y: np.ndarray, k_indices: np.ndarray, k_th_fold: int
    ) -> Union[float, tuple[float, float]]:
        """Evaluate the model on a given fold"""
        # Create a mask to select the train data
        mask = np.ones((self.k_folds), dtype=bool)
        mask[k_th_fold] = False
        train_X, train_y = X[k_indices[mask].flatten()], y[k_indices[mask].flatten()]
        test_X, test_y = X[k_indices[k_th_fold]], y[k_indices[k_th_fold]]

        self.model.fit(train_X, train_y)
        return (
            self.metric(self.model.predict(test_X), test_y),
            self.metric(self.model.predict(train_X), train_y),
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model on all folds

        Parameters
        ----------
        X : np.ndarray
            The data
        y : np.ndarray
            The labels

        Returns
        -------
        dict
            A dictionary containing the averaged scores and the scores per fold for both the train and test sets
        """
        k_indices = self.build_k_indices(y)
        test_train_scores_per_fold = np.array(
            [
                self._evaluate_fold(X, y, k_indices, k_th_fold)
                for k_th_fold in range(self.k_folds)
            ]
        )  # Dimension: (k_folds, 2) where the second dimension is (test_score, train_score)
        return {
            "test_scores_averaged": np.mean(test_train_scores_per_fold[:, 0]),
            "test_scores_per_fold": test_train_scores_per_fold[:, 0].T,
            "train_scores_averaged": np.mean(test_train_scores_per_fold[:, 1]),
            "train_scores_per_fold": test_train_scores_per_fold[:, 1].T,
        }


class StratifiedUpsampledCV:
    def __init__(self, n_splits=5, random_state=1):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y):
        np.random.seed(self.random_state)
        # Get the indices of the positive and negative samples
        min_idxs = np.argwhere(y == 1).squeeze()
        maj_idxs = np.argwhere(y == -1).squeeze()

        # Shuffle the indices
        np.random.shuffle(min_idxs)
        np.random.shuffle(maj_idxs)

        # Determine the number of samples to drop from the end of the indices
        CV_min_drop = int(self.n_splits * (min_idxs.shape[0] // self.n_splits))
        CV_maj_drop = int(self.n_splits * (maj_idxs.shape[0] // self.n_splits))
        min_idxs = min_idxs[:CV_min_drop]
        maj_idxs = maj_idxs[:CV_maj_drop]

        # Now split the indices into n_splits
        min_idxs = min_idxs.reshape((self.n_splits, -1))
        maj_idxs = maj_idxs.reshape((self.n_splits, -1))

        for i in range(self.n_splits):
            mask = np.ones(self.n_splits, dtype=bool)
            mask[i] = False
            train_maj_idxs = maj_idxs[mask].flatten()
            train_min_idxs = min_idxs[mask].flatten()
            n_dup = np.ceil(train_maj_idxs.shape[0] / train_min_idxs.shape[0]).astype(
                int
            )
            train_min_idxs = np.tile(train_min_idxs, n_dup).squeeze()
            train_min_idxs = np.random.choice(
                train_min_idxs, size=train_maj_idxs.shape[0], replace=False
            ).squeeze()
            train_idxs = np.concatenate((train_min_idxs, train_maj_idxs))
            np.random.shuffle(train_idxs)
            test_idxs = np.concatenate((maj_idxs[i], min_idxs[i]))

            yield train_idxs, test_idxs


if __name__ == "__main__":
    import timeit
    from copy import deepcopy

    import matplotlib.pyplot as plt
    import numpy as np

    from base_regressors import *
    from cross_validation import CrossValidation
    from enhanced_regressors import *
    from grid_search import GridSearch
    from metrics import F1, F1_continuous
    from pre_processing import *
    from utils import create_csv_submission, load_data, upsample_data

    X_train, X_test, y_train, X_cols, train_idxs, test_idxs = load_data()
    X_train.shape, y_train.shape, len(X_cols), X_test.shape

    preprocessor_pipe = Pipeline(
        steps=[
            (FeatureDropper, None),
            (ToNaNFiller, None),
            (MissingValuesDropper, None),
            (ConstantFeaturesRemover, None),
            (FillImputer, {"strategy": "median"}),
            (PropPredictorsRemover, None),
            (Standardizer, {"with_nan": False}),
            (OneHotEncoder, {"drop_last": True}),
        ],
    )
    print(f"{X_train.shape=}")
    start_preprocessing = timeit.default_timer()
    X_train, X_train_cols = preprocessor_pipe.fit_transform(X_train, deepcopy(X_cols))
    end_preprocessing = timeit.default_timer()
    print(f"{X_train.shape=}")
    print(f"{end_preprocessing-start_preprocessing=}")

    from enhanced_regressors import LassoRegressor

    LR = LassoRegressor(
        lambda_=0.05, gamma=0.5, max_iters=10, batch_size=None, intercept=False
    )
    CV = CrossValidation(LR, 5, F1)
    test_score_per_fold = CV.evaluate(X_train, y_train)
    LR.get_weights()

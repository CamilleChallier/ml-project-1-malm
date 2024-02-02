import pickle
from typing import Union

import numpy as np

from base_regressors import *
from clustering import KMeans
from metrics import F1, accuracy
from pre_processing import PolynomialFeatures
from utils import batch_iter


class PolynomialRegressor(Regressor):
    """Polynomial wrapper for a regressor"""

    def __init__(
        self,
        degree: int,
        X_cols: list[str],
        regressor_class,
        **regressor_kwargs,
    ) -> None:
        """
        Initialize the regressor.


        Parameters
        ----------
        degree : int
            Degree of the polynomial
        X_cols : list[str]
            List of the columns of the feature matrix
        regressor_class
            The type of regressor to use
        **regressor_kwargs
            The kwargs to pass to the regressor
        """
        self.degree = degree
        self.X_cols = X_cols
        self.regressor = regressor_class(**regressor_kwargs)
        self.polynomial_builder = PolynomialFeatures(self.degree)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the regressor

        Parameters
        ----------
        X : np.ndarray
            The training feature matrix
        y : np.ndarray
            The target values

        Returns
        -------
        PolynomialRegressor
            The fitted regressor
        """
        X_poly, X_cols_w_poly = self.polynomial_builder.fit_transform(X, self.X_cols)
        self.X_cols_w_poly = X_cols_w_poly
        self.regressor.fit(X_poly, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target values

        Parameters
        ----------
        X : np.ndarray
            The test feature matrix

        Returns
        -------
        np.ndarray
            The predicted target values
        """
        X_poly, X_cols_w_poly = self.polynomial_builder.transform(X, self.X_cols)
        assert self.X_cols_w_poly == X_cols_w_poly
        return self.regressor.predict(X_poly)


class LassoPenalty:
    """
    Lasso penalty.
    """

    def __init__(self, lambda_):
        """
        Parameters
        ----------
        lambda_ : float
            Regularization parameter.
        """
        self.lambda_ = lambda_

    def calculate_penalty(self, w):
        return self.lambda_ * np.sum(np.abs(w))

    def derivate(self, w):
        return self.lambda_ * np.sign(w)


class ElasticPenalty:
    """
    ElasticNet penalty.
    """

    def __init__(self, lambda_, l_ratio):
        """
        Parameters
        ----------
        lambda_ : float
            Regularization parameter.
        l_ratio : float
            Ratio between the L1 and L2 regularization.
        """
        self.lambda_ = lambda_
        self.lambda_ratio = l_ratio

    def calculate_penalty(self, w):
        """
        Compute the norm

        Parameters
        ----------
        w : numpy.ndarray
            Weights.

        Returns
        -------
        float
            Norm.
        """
        l1_contribution = self.lambda_ratio * self.lambda_ * np.sum(np.abs(w))
        l2_contribution = (
            (1 - self.lambda_ratio) * self.lambda_ * 0.5 * np.sum(np.square(w))
        )
        return l1_contribution + l2_contribution

    def derivate(self, w):
        """
        Compute the derivation of the norm

        Parameters
        ----------
        w : numpy.ndarray
            Weights.
        """
        l1_derivation = self.lambda_ * self.lambda_ratio * np.sign(w)
        l2_derivation = self.lambda_ * (1 - self.lambda_ratio) * w
        return l1_derivation + l2_derivation


class PenalyzedRegressor(Regressor):
    """
    Base class for penalized regressors Lasso and ElasticNet.
    """

    def __init__(
        self,
        penalty: Union[LassoPenalty, ElasticPenalty],
        lambda_: float,
        gamma: float,
        path: bool = False,
        intercept: bool = True,
        max_iters: int = 100,
        initial_w: np.ndarray = None,
        batch_size: int = None,
        seed: int = 1,
    ) -> None:
        """
        Initialize the regressor.

        Parameters
        ----------
        penalty : Union[LassoPenalty, ElasticPenalty]
            Penalty to be used.
        lambda_ : float
            Regularization parameter.
        gamma : float
            Learning rate.
        path : bool, optional
            Whether to use the path algorithm or not, by default False
        intercept : bool, optional
            Whether to use an intercept or not, by default True
        max_iters : int, optional
            Maximum number of iterations, by default 100
        initial_w : numpy.ndarray, optional
            Initial weights, by default None
        batch_size : int, optional
            Batch size for stochastic gradient descent, by default None
        seed : int, optional
            Random seed, by default 1
        """
        self.penalty = penalty
        self.lambda_ = lambda_
        self.gamma = gamma
        self.max_iters = max_iters
        self.initial_w = initial_w
        self.path = path
        self.intercept = intercept
        self.save_loss = []
        self.batch_size = batch_size
        self.seed = seed

    def soft_threshold(self, rho):
        """
        Soft thresholding operator

        Parameters
        ----------
        rho : float
            Input.

        Returns
        -------
        float
            Output.
        """
        if rho < -self.lambda_:
            return rho + self.lambda_
        elif rho > self.lambda_:
            return rho - self.lambda_
        else:
            return 0

    def compute_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function

        Parameters
        ----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns
        -------
        numpy.ndarray
            Gradient of the loss function.
        """
        return -1 / y.shape[0] * X.T.dot(y - X.dot(self.w)) + self.penalty.derivate(
            self.w
        )

    def coordinate_descent_lasso(self, X, y):
        """
        Coordinate descent for Lasso regression

        Parameters
        ----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns
        -------
        numpy.ndarray
            Updated weights.
        """
        # Looping through each coordinate
        for j in range(X.shape[1]):
            # Vectorized implementation
            X_j = X[:, j].reshape(-1, 1)
            y_pred = X.dot(self.w)
            rho = X_j.T.dot(y - y_pred + self.w[j] * X_j)

            self.w[j] = self.soft_threshold(rho)
        return self.w

    def calculate_loss(self, X, y, n):
        """
        Compute the loss function

        Parameters
        ----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns
        -------
        float
            Loss function.
        """
        return (1 / (2 * n)) * np.sum(
            np.square(X.dot(self.w) - y)
        ) + self.penalty.calculate_penalty(self.w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        if self.intercept == True:
            X = np.insert(X, 0, 1, axis=1)

        if self.initial_w is None:
            self._initialise_w(X.shape[1])
        self.w = self.initial_w

        if self.batch_size is None:
            for _ in range(self.max_iters):
                self.train_iter(X, y)
        else:
            for minibatch_X, minibatch_y in batch_iter(
                X, y, self.batch_size, self.max_iters, seed=self.seed
            ):
                self.train_iter(minibatch_X, minibatch_y)

        return self

    def train_iter(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model for one iteration.

        Parameters
        ----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.
        """
        if self.path:
            self.w = self.coordinate_descent_lasso(X, y)
        else:
            self.w -= self.gamma * self.compute_gradient(X, y)
        self.save_loss.append(self.calculate_loss(X, y, X.shape[0]))


class LassoRegressor(PenalyzedRegressor):
    """
    Lasso regressor.
    """

    def __init__(
        self,
        lambda_: float,
        gamma: float,
        path: bool = False,
        intercept: bool = True,
        max_iters: int = 100,
        initial_w=None,
        batch_size: int = None,
        seed: int = 1,
    ):
        """
        Parameters
        ----------
        lambda_ : float
            Regularization parameter.
        gamma : float
            Learning rate.
        path : bool, optional
            Whether to use the path algorithm or not, by default False
        intercept : bool, optional
            Whether to use an intercept or not, by default True
        max_iters : int, optional
            Maximum number of iterations, by default 100
        initial_w : numpy.ndarray, optional
            Initial weights, by default None
        batch_size : int, optional
            Batch size for stochastic gradient descent, by default None
        seed : int, optional
            Random seed, by default 1
        """
        super().__init__(
            LassoPenalty(lambda_),  # penalty
            lambda_,
            gamma,
            path,
            intercept,
            max_iters,
            initial_w,
            batch_size,
            seed,
        )


class ElasticNet(PenalyzedRegressor):
    """
    ElasticNet regressor.
    """

    def __init__(
        self,
        l_ratio: float,
        lambda_: float,
        gamma: float,
        path: bool = False,
        intercept: bool = True,
        max_iters: int = 100,
        initial_w=None,
        batch_size: int = None,
        seed: int = 1,
    ):
        """
        Parameters
        ----------
        l_ratio : float
            Ratio between the L1 and L2 regularization.
        lambda_ : float
            Regularization parameter.
        gamma : float
            Learning rate.
        path : bool, optional
            Whether to use the path algorithm or not, by default False
        intercept : bool, optional
            Whether to use an intercept or not, by default True
        max_iters : int, optional
            Maximum number of iterations, by default 100
        initial_w : numpy.ndarray, optional
            Initial weights, by default None
        batch_size : int, optional
            Batch size for stochastic gradient descent, by default None
        seed : int, optional
            Random seed, by default 1
        """
        super().__init__(
            ElasticPenalty(lambda_, l_ratio),
            lambda_,
            gamma,
            path,
            intercept,
            max_iters,
            initial_w,
            batch_size,
            seed,
        )


class ImbalancedLogisticRegressor(LogisticRegressor):
    """Imbalanced logistic logistic regression as described in the paper:
    Zhang, Lili et al. “Improving logistic regression on the
    imbalanced data by a novel penalized log-likelihood function.”
    Journal of applied statistics vol. 49,13 3257-3277.
    16 Jun. 2021, doi:10.1080/02664763.2021.1939662
    """

    def __init__(
        self,
        gamma: float,
        alpha: float,
        max_iters: int = 100,
        initial_w: np.ndarray = None,
    ) -> None:
        """
        Imbalanced logistic regression

        Parameters
        ----------
        gamma : float
            Learning rate
        alpha : float
            Learning rate for the penalty term p (lambda in the paper)
        max_iters : int, optional
            Maximum number of weight updates, by default 100
        initial_w : np.ndarray, optional
            The initial weights, by default None
        """
        super().__init__(initial_w, max_iters, gamma)
        self.gamma = gamma
        self.alpha = alpha

    def calculate_weight_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the weights

        Parameters
        ----------
        X : np.ndarray
            The feature matrix
        y : np.ndarray
            The target values

        Returns
        -------
        np.ndarray
            The gradient of the weights
        """
        y_pred = self.sigmoid(np.dot(X, self.w))
        return (
            1 / X.shape[0] * np.dot(X.T, y_pred - y * (self.p + y_pred * (1 - self.p)))
        )

    def calculate_p_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the penalty term p

        Parameters
        ----------
        X : np.ndarray
            The feature matrix
        y : np.ndarray
            The target values

        Returns
        -------
        np.ndarray
            The gradient of the penalty term p
        """
        return y * np.log(1 + np.exp(X.dot(self.w)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the regressor

        Parameters
        ----------
        X : np.ndarray
            The feature matrix
        y : np.ndarray
            The target values

        Returns
        -------
        ImbalancedLogisticRegressor
            The fitted regressor
        """
        if self.initial_w is None:
            self._initialise_w(X.shape[1])
        y_0_1 = np.where(y == -1, 0, y)
        self.w = self.initial_w

        self.p = np.ones((X.shape[0],))

        for _ in range(self.max_iters):
            self.w -= self.gamma * self.calculate_weight_gradient(X, y_0_1)
            self.p += self.alpha * self.calculate_p_gradient(X, y_0_1)
        return self


class LLRENSRegressor(Regressor):
    """LLRENSRegressor as described in the paper:
    Wang H, Xu Q, Zhou L (2015)
    Large Unbalanced Credit Scoring Using Lasso-Logistic Regression Ensemble.
    PLoS ONE 10(2): e0117844. https://doi.org/10.1371/journal.pone.0117844
    """

    def __init__(
        self,
        majority_clustering_groups: np.ndarray,
        lambdas: list[float],
        gammas: list[float],
        max_iters: list[float],
        majority_class: int = -1,
        minority_class: int = 1,
        seed: int = 1,
    ) -> None:
        """LLRENSRegressor

        Parameters
        ----------
        majority_clustering_groups : np.ndarray
            The indexes of the clusters of the majority class
        lambdas : list[float]
            The lambdas for each LLR
        gammas : list[float]
            The gammas for each LLR
        max_iters : list[float]
            The maximum number of iterations for each LLR
        majority_class : int, optional
            The value of the majority class, by default -1
        minority_class : int, optional
            The value of the minority class, by default 1
        seed : int, optional
            The random seed, by default 1
        """
        majority_clustering_groups = majority_clustering_groups.reshape(-1, 1)
        self.seed = seed
        self.LLRs = [
            LogisticRegressor(
                initial_w=None,
                max_iters=max_iter,
                gamma=gamma,
                lambda_=lambda_,
                learning="gradient_descent",
                reg_norm="l1",
                batch_size=None,
                seed=self.seed,
            )
            for max_iter, gamma, lambda_ in zip(max_iters, gammas, lambdas)
        ]
        self.minority_class = minority_class
        self.majority_class = majority_class

        # Make the smallest majority cluster index += the minority class +1
        # (add +1 to avoid case majority cluster index 0 + minority class 1 = 1 which is the minority class)
        majority_clustering_groups += self.minority_class + 1

        self.majority_clustering_groups = majority_clustering_groups

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold_quantile: float = 0.0,
    ):
        """Fit the LLRENSRegressor

        Parameters
        ----------
        X : np.ndarray
            The feature matrix
        y : np.ndarray
            The target values
        threshold_quantile : float, optional
            Only subgroups which have sizes larger than a threshold quantile of the minority class are retained, by default 0.0
        """
        # Create a group index for each cluster in the majority and the minority class
        self.group_indexes = self.minority_class * np.ones_like(y)

        # Set the group indexes of the majority class to the cluster indexes
        self.group_indexes[
            np.argwhere(y == self.majority_class)
        ] = self.majority_clustering_groups

        # Keep only the majority clusters that have data greater or equal to 75% of the minority class
        self.maj_subgroups = [
            cluster_idx
            for cluster_idx in np.unique(self.majority_clustering_groups)
            if np.sum(self.group_indexes == cluster_idx)
            >= threshold_quantile * np.sum(y == self.minority_class)
        ]

        print(f"Fitting {len(self.LLRs)} LLRs ...")
        self.voting_weights = []
        for i, maj_subgroup_idx, LLR in zip(
            range(len(self.LLRs)), self.maj_subgroups, self.LLRs
        ):
            print(f"Fitting LLR {i+1}/{len(self.LLRs)}")
            print(
                f"\tmax_iters_{i}={LLR.max_iters}\n\tgamma_{i}={LLR.gamma}\n\tlambda_{i}={LLR.lambda_}"
            )
            training_idxs = self._bootstrap_minority_data(maj_subgroup_idx)
            testing_mask = self.group_indexes != maj_subgroup_idx
            LLR.fit(X[training_idxs], y[training_idxs])
            score = F1(LLR.predict(X[testing_mask]), y[testing_mask])
            self.voting_weights.append(1 / (1 + np.exp(-score)))
            print(f"\tscore_{i}={score}")
        return

    def predict(
        self, X: np.ndarray, method: str = "percentage", percentage: float = 0.3
    ) -> np.ndarray:
        """Predict the target values

        Parameters
        ----------
        X : np.ndarray
            The feature matrix
        method : str, optional
            The voting method to use, either `weight_voting` or `percentage`, by default "percentage"
        percentage : float, optional
            The percentage of agreement to categorize an event, only used if method is set to `percentage`, by default 0.3

        Returns
        -------
        np.ndarray
            The predicted target values

        Raises
        ------
        ValueError
            If the method is not implemented
        """
        if method == "weight_voting":
            # As in the paper
            pred = [
                voting_weight * LLR.predict(X)
                for voting_weight, LLR in zip(self.voting_weights, self.LLRs)
            ]
            pred = np.sum(pred, axis=0)
            pred = np.sign(pred)
        elif method == "percentage":
            # If a certain percentage of LLRs agree that it is a 1
            pred = [LLR.predict(X) for LLR in self.LLRs]
            pred = np.where(pred == -1, 0, pred)
            pred = np.sum(pred, axis=0)
            pred = np.where(pred > percentage * len(self.LLRs), 1, -1)
        else:
            raise ValueError("Method `{mehtod}` for predicting is not implemented")
        return pred

    def _bootstrap_minority_data(self, majority_subgroup_idx: int) -> np.ndarray:
        """Bootstrap the minority data"""
        minority_indexes = np.argwhere(
            self.group_indexes == self.minority_class
        ).squeeze()
        cluster_indexes = np.argwhere(
            self.group_indexes == majority_subgroup_idx
        ).squeeze()
        np.random.seed(self.seed)
        bootstrap_indexes = np.random.choice(
            minority_indexes, size=cluster_indexes.shape[0], replace=True
        )
        return np.concatenate((cluster_indexes, bootstrap_indexes))


if __name__ == "__main__":
    import pickle
    import timeit
    from copy import deepcopy

    import matplotlib.pyplot as plt
    import numpy as np

    from clustering import KMeans
    from pre_processing import *
    from utils import create_csv_submission, load_data

    X_train, X_test, y_train, X_cols, train_idxs, test_idxs = load_data()

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

    kmeans = KMeans(n_clusters=10, max_iter=100)
    # Please run the kmeans.ipynb notebook to generate the centroids
    with open("kmeans_10c_m1_data.pkl", "rb") as f:
        kmeans.centroids, kmeans.last_centroid_idxs = pickle.load(f)

    print(f"{y_train.shape=}")

    llre = LLRENSRegressor(
        majority_clustering_groups=kmeans.last_centroid_idxs,
        # gammas=np.ones((10,)) * 0.01,
        # lambdas=np.ones((10,)) * 0,
        # max_iters = np.ones((10,)).astype(int) * 200,
        gammas=[
            0.0004138804388474671,
            4.340636234977542e-06,
            10.0,
            1e-06,
            1.1435195639003527e-06,
            1e-06,
            0.015105047115050384,
            1.5314619982277378e-06,
            1e-06,
            7.535722405143262e-05,
        ],
        lambdas=[
            0.12807852581681167,
            1.0500928541079033e-06,
            2.419255029022902,
            1e-06,
            1e-06,
            3.14793289174295e-05,
            9.3465204245773,
            9.918920664897113,
            10,
            1.139221842206064e-05,
        ],
        max_iters=[
            450,
            156,
            195,
            70,
            10,
            10,
            491,
            37,
            37,
            500,
        ],
    )
    print("Fitting LLRE")
    llre.fit(X_train, y_train)

    print("Predicting")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.05), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.10), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.15), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.20), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.25), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.30), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.35), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.40), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.45), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.50), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.55), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.60), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.65), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.70), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.75), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.80), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.85), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.90), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 0.95), y_train)=}")
    # print(f"{F1(llre.predict(X_train, 'percentage', 1.00), y_train)=}")

    # print(f"{F1(llre.predict(X_train, 'weight_voting'), y_train)=}")
    print(f"{F1(llre.predict(X_train, 'percentage', 0.3), y_train)=}")
    print(f"{accuracy(llre.predict(X_train, 'percentage', 0.4), y_train)=}")

    X_test, X_test_cols = preprocessor_pipe.transform(X_test, deepcopy(X_cols))
    # y_test_pred = llre.predict(X_test, 'weight_voting')
    y_test_pred = llre.predict(X_test, "percentage", 0.3)
    create_csv_submission(test_idxs, y_test_pred, "llre_v1_0.csv")

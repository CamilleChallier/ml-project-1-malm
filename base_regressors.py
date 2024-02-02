from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from metrics import F1, mae, mse, rmse
from utils import batch_iter


class Regressor(ABC):
    """
    Abstract base class for regression models.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initialise the regressor.
        """
        self.w = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.

        Returns
        -------
        numpy array of shape (N,) : Predictions.
        """
        if self.w is None:
            raise ValueError(
                "Weights have not been calculated yet. Use `fit` method beforehand."
            )
        return np.dot(X, self.w)

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Mean absolute error.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        float : Mean absolute error.
        """
        return mae(self.predict(X), y)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Mean squared error.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        float : Mean squared error.
        """
        return mse(self.predict(X), y)

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Root mean squared error.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        float : Root mean squared error.
        """
        return rmse(self.predict(X), y)

    def get_weights(self) -> np.ndarray:
        """
        Get the weights of the regressor.
        """
        if self.w is None:
            raise ValueError(
                "Weights have not been calculated yet. Use `fit` method beforehand."
            )
        return self.w

    def _initialise_w(self, d: int) -> None:
        """
        Initialise the weights of the regressor.

        Parameters
        ----------
        d : int
            The number of features.

        Returns
        -------
        None
        """
        if not hasattr(self, "initial_w"):
            raise ValueError("Regressor does not have an attribute `initial_w`.")
        self.initial_w = np.zeros((d,))


class LeastSquaresRegressor(Regressor):
    """
    Least squares regression model.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model using least squares. Compute the weights.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        self : object
        """
        self.w, _, _, _ = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, y), rcond=None)
        return self


class RidgeRegressor(Regressor):
    """
    Ridge regression model.
    """

    def __init__(self, lambda_: float) -> None:
        """
        Initialise the regressor.

        Parameters
        ----------
        lambda_ : float
            The regularization parameter.
        """
        self.lambda_ = lambda_

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model using ridge regression. Compute the weights.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        self : object
        """
        self.w, _, _, _ = np.linalg.lstsq(
            np.dot(X.T, X) + (self.lambda_ * 2 * X.shape[0]) * np.eye(X.shape[1]),
            np.dot(X.T, y),
            rcond=None,
        )
        return self


class GDRegressor(Regressor):
    """
    Linear regression model using gradient descent.
    """

    def __init__(
        self,
        gamma: float,
        max_iters: int = 100,
        loss_kind: str = "mse",
        initial_w: np.ndarray = None,
        batch_size: int = None,
        seed: int = 1,
    ) -> None:
        """
        Initialise the regressor.

        Parameters
        ----------
        gamma : float
            The learning rate.
        max_iters : int, optional
            The maximum number of iterations, by default 100
        loss_kind : str, optional
            The loss function to be used, by default "mse"
        initial_w : np.ndarray, optional
            The initial weights, by default None. If none, the weights are initialised to zero.
        batch_size : int, optional
            The batch size, by default None. If none, the whole dataset is used.
        seed : int, optional
            The seed to be used for the random number generator, by default 1
        """
        self.gamma = gamma
        self.max_iters = max_iters
        self.loss_kind = loss_kind
        self.initial_w = initial_w
        self.batch_size = batch_size
        self.seed = seed

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Fit the model using gradient descent. Compute the weights.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        self : object
        """
        if self.initial_w is None:
            self._initialise_w(X.shape[1])
        self.w = deepcopy(self.initial_w)
        if self.batch_size is None:
            for _ in range(self.max_iters):
                self.w -= self.gamma * self.compute_gradient(X, y)
        else:
            for X, y in batch_iter(
                X, y, self.batch_size, self.max_iters, seed=self.seed
            ):
                self.w -= self.gamma * self.compute_gradient(X, y)
        return self

    def compute_gradient(self, X, y) -> np.ndarray:
        """
        Compute the gradient, call the methods for the different loss functions.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D,) : The gradient.
        """
        if self.loss_kind == "mse":
            return self.compute_gradient_mse(X, y)
        elif self.loss_kind == "mae":
            return self.compute_subgradient_mae(X, y)
        else:
            raise ValueError("The error should be either `mae` or `mse`.")

    def compute_loss(self, X, y) -> np.ndarray:
        """
        Compute the loss.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D,) : The loss.
        """
        if self.loss_kind == "mse":
            return self.mse(X, y)
        elif self.loss_kind == "mae":
            return self.mae(X, y)
        else:
            raise ValueError("The error should be either `mae` or `mse`.")

    def compute_gradient_mse(self, X, y):
        """
        Compute the gradient using the mean squared error.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D,) : The mse gradient.
        """
        return -1 / y.shape[0] * X.T.dot(y - X.dot(self.w))

    def compute_subgradient_mae(self, X, y):
        """
        Compute the gradient using the mae.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D,) : The mae gradient.
        """
        return -1 / y.shape[0] * X.T.dot(np.sign(y - X.dot(self.w)))


class LogisticRegressor(Regressor):
    """
    Logistic regression model.
    """

    def __init__(
        self,
        initial_w: np.ndarray = None,
        max_iters: int = 300,
        gamma: float = 0.5,
        lambda_: float = 0.0,
        learning: str = "gradient_descent",
        reg_norm: str = "l2",
        batch_size: int = None,
        seed: int = 1,
    ) -> None:
        """
        Initialise the regressor.

        Parameters
        ----------
        initial_w : np.ndarray, optional
            The initial weights, by default None. If none, the weights are initialised to zero.
        max_iters : int, optional
            The maximum number of iterations, by default 100
        gamma : float, optional
            The learning rate, by default 0.5
        lambda_ : float, optional
            The regularization parameter, by default 0.0
        learning : str, optional
            The learning method, by default "gradient_descent"
        reg_norm : str, optional
            The regularization norm, by default "l2"
        batch_size : int, optional
            The batch size, by default None. If none, the whole dataset is used.
        seed : int, optional
            The seed to be used for the random number generator, by default 1
        """
        self.initial_w = initial_w
        self.max_iters = max_iters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.learning = learning
        self.batch_size = batch_size
        self.seed = seed
        self.reg_norm = reg_norm

    def sigmoid(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Sigmoid function.

        Parameters
        ----------
        y_pred : numpy array of shape (N,)
            The predictions.

        Returns
        -------
        numpy array of shape (N,) : The sigmoid of the predictions.
        """
        return 1 / (1 + np.exp(-y_pred))

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the log likelihood.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (N,) : The log likelihood.
        """
        return (
            -1
            / (X.shape[0])
            * np.sum(
                y * np.log(self.sigmoid(X.dot(self.w)))
                + (1 - y) * np.log(1 - self.sigmoid(X.dot(self.w)))
            )
        )

    def calculate_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D,) : The gradient.
        """
        return X.T.dot(self.sigmoid(X.dot((self.w))) - y) / X.shape[0]

    def calculate_hessian(self, X: np.ndarray):
        """
        Compute the hessian.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D, D) : The hessian.
        """
        return (
            1
            / X.shape[0]
            * X.T.dot(
                np.diag(
                    (
                        self.sigmoid(X.dot(self.w)) * (1 - self.sigmoid(X.dot(self.w)))
                    ).flatten()
                )
            ).dot(X)
        )

    def calculate_newton_gradient(self, X: np.ndarray, y: np.ndarray):
        """
        Compute the gradient using the Newton method.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D,) : The gradient using the Newton method.
        """
        return np.linalg.lstsq(
            self.calculate_hessian(X),
            self.calculate_gradient(X, y)
            if self.lambda_ == 0
            else self.calculate_penalized_gradient(X, y),
            rcond=None,
        )

    def learning_by_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """
        Compute the gradient.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D,) : The gradient.
        """
        return self.calculate_gradient(X, y)

    def calculate_penalized_gradient(self, X: np.ndarray, y: np.ndarray):
        """
        Compute the gradient with the regularization term. Could be either `l1` or `l2`.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        numpy array of shape (D,) : The penalized gradient.
        """
        if self.reg_norm == "l2":
            return self.calculate_gradient(X, y) + self.lambda_ * self.w * 2
        elif self.reg_norm == "l1":
            return self.calculate_gradient(X, y) + self.lambda_ * np.sign(self.w)
        else:
            raise ValueError("The regularization norm should be either `l1` or `l2`.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model using gradient descent. Compute the weights.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            Feature matrix, where N is the number of samples and D is the number of features.
        y : numpy array of shape (N,)
            Labels.

        Returns
        -------
        self : object
        """
        if self.initial_w is None:
            self._initialise_w(X.shape[1])
        y_0_1 = np.where(y == -1, 0, y)
        self.w = deepcopy(self.initial_w)
        if self.batch_size is None:
            for _ in range(self.max_iters):
                self.train_iter(X, y_0_1)
        else:
            for minibatch_X, minibatch_y in batch_iter(
                X, y_0_1, self.batch_size, self.max_iters, seed=self.seed
            ):
                self.train_iter(minibatch_X, minibatch_y)

        return self

    def train_iter(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model for one iteration.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            The data.
        y : numpy array of shape (N,)
            The labels.
        """
        if self.learning == "gradient_descent":
            if self.lambda_ == 0:
                self.w -= self.gamma * self.calculate_gradient(X, y)
            else:
                self.w -= self.gamma * self.calculate_penalized_gradient(X, y)
        elif self.learning == "newton":
            self.w -= self.gamma * self.calculate_newton_gradient(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the logistic model.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            The data.

        Returns
        -------
        numpy array of shape (N,) : The predictions.
        """
        return np.sign(super().predict(X))

    def get_params(self, deep: bool = True) -> dict:
        """
        Get the parameters of the model.

        Parameters
        ----------
        deep : bool, optional
            Whether to recursively get the parameters, by default True

        Returns
        -------
        dict : The parameters of the model.
        """
        return {
            "gamma": self.gamma,
            "max_iters": self.max_iters,
            "lambda_": self.lambda_,
        }

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the F1 score of the model.

        Parameters
        ----------
        X : numpy array of shape (N, D)
            The data.
        y : numpy array of shape (N,)
            The labels.

        Returns
        -------
        float : The F1 score.
        """
        score = F1(self.predict(X), y)
        if np.isnan(score):
            score = 0.0
        print(f"{score=}")
        return score

    def set_params(self, **parameters):
        """
        Set the parameters of the model.

        Parameters
        ----------
        **parameters : dict
            The parameters to set.

        Returns
        -------
        self : object
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


if __name__ == "__main__":
    import timeit

    from metrics import F1
    from pre_processing import *
    from utils import downsample_data, load_data

    custom_F1 = lambda y_pred, y_true: F1(np.sign(y_pred), y_true)

    X_train, X_test, y_train, X_cols, _, _ = load_data()
    # X_train, y_train = downsample_data(X_train, y_train)

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

    start_fitting = timeit.default_timer()

    model = LogisticRegressor(gamma=0.1, max_iters=100).fit(X_train, y_train)

    end_fitting = timeit.default_timer()
    print(f"{end_fitting-start_fitting=}")

    print(f"{F1(model.predict(X_train), y_train)}")

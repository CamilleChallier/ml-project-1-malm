import numpy as np

from base_regressors import (
    GDRegressor,
    LeastSquaresRegressor,
    LogisticRegressor,
    RidgeRegressor,
)


def mean_squared_error_gd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> tuple[np.ndarray, float]:
    """Mean squared error regression using gradient descent

    Parameters
    ----------
    y : np.ndarray
        The target values of shape (N,) where N is the number of samples
    tx : np.ndarray
        The feature matrix of shape (N, D) where D is the number of features
    initial_w : np.ndarray
        The initial weights of shape (D,)
    max_iters : int
        The maximum number of iterations
    gamma : float
        The learning rate

    Returns
    -------
    tuple[np.ndarray, float]
        The optimal weights of shape (D,) and the mean squared error
    """
    GD = GDRegressor(
        gamma=gamma, max_iters=max_iters, loss_kind="mse", initial_w=initial_w
    ).fit(tx, y)
    return GD.get_weights(), GD.compute_loss(tx, y)


def mean_squared_error_sgd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> tuple[np.ndarray, float]:
    """Mean squared error regression using stochastic gradient descent

    Parameters
    ----------
    y : np.ndarray
        The target values of shape (N,) where N is the number of samples
    tx : np.ndarray
        The feature matrix of shape (N, D) where D is the number of features
    initial_w : np.ndarray
        The initial weights of shape (D,)
    max_iters : int
        The maximum number of iterations
    gamma : float
        The learning rate

    Returns
    -------
    tuple[np.ndarray, float]
        The optimal weights of shape (D,) and the mean squared error
    """
    SGD = GDRegressor(
        gamma=gamma,
        batch_size=1,
        max_iters=max_iters,
        loss_kind="mse",
        initial_w=initial_w,
    ).fit(tx, y)
    return SGD.get_weights(), SGD.compute_loss(tx, y)


def least_squares(y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculate the least squares solution.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,), where N is the number of samples.
    tx : np.ndarray
        Feature matrix of shape (N, D), where D is the number of features.

    Returns
    -------
    tuple[np.ndarray, float]
        The optimal weights of shape (D,) and the mean squared error

    Examples
    --------
    >>> import numpy as np
    >>> least_squares(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    least_squares_regressor = LeastSquaresRegressor().fit(tx, y)
    return least_squares_regressor.get_weights(), least_squares_regressor.mse(tx, y)


def ridge_regression(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> tuple[np.ndarray, float]:
    """
    Calculate the ridge regression solution.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,), where N is the number of samples.
    tx : np.ndarray
        Feature matrix of shape (N, D), where D is the number of features.
    lambda_ : float
        Regularization parameter.

    Returns
    -------
    tuple[np.ndarray, float]
        The optimal weights of shape (D,) and the mean squared error

    Examples
    --------
    >>> import numpy as np
    >>> ridge_regression(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    ridge_regressor = RidgeRegressor(lambda_=lambda_).fit(tx, y)
    return ridge_regressor.get_weights(), ridge_regressor.mse(tx, y)


def logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Logistic regression using gradient descent

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,), where N is the number of samples.
    tx : np.ndarray
        Feature matrix of shape (N, D), where D is the number of features.
    initial_w : np.ndarray
        The initial weights of shape (D,).
    max_iters : np.ndarray
        The maximum number of iterations.
    gamma : np.ndarray
        The learning rate.

    Returns
    -------
    tuple[np.ndarray, float]
        The optimal weights of shape (D,) and the log likelihood.
    """
    logistic_regressor = LogisticRegressor(
        initial_w=initial_w, max_iters=max_iters, gamma=gamma
    ).fit(tx, y)
    return logistic_regressor.get_weights(), logistic_regressor.log_likelihood(tx, y)


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """Regularised logistic regression using gradient descent

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (N,), where N is the number of samples.
    tx : np.ndarray
        Feature matrix of shape (N, D), where D is the number of features.
    lambda_ : float
        Regularization parameter used for the penalization of the weights.
    initial_w : np.ndarray
        The initial weights of shape (D,).
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The learning rate.

    Returns
    -------
    tuple[np.ndarray, float]
        The optimal weights of shape (D,) and the log likelihood.
    """
    logistic_regressor = LogisticRegressor(
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
        lambda_=lambda_,
        reg_norm="l2",
    ).fit(tx, y)
    return logistic_regressor.get_weights(), logistic_regressor.log_likelihood(tx, y)


if __name__ == "__main__":
    expected_w = np.array([0.413044, 0.875757])
    y = np.array([0.1, 0.3, 0.5])
    tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
    w_init = np.array([0.5, 1.0])

    expected_w = np.array([0.413044, 0.875757])
    expected_loss = 2.959836

    print(mean_squared_error_gd(y, tx, expected_w, 0, 0.1))

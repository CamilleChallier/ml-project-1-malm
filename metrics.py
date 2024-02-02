import numpy as np


def mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y : np.ndarray
        The true values.

    Returns
    -------
    float
        The mean squared error.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> mse(np.random.rand(100), np.random.rand(100))
    0.0956541466988608
    """
    return 0.5 * np.mean((y_pred - y) ** 2)


def rmse(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y : np.ndarray
        The true values.

    Returns
    -------
    float
        The root mean squared error.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> rmse(np.random.rand(100), np.random.rand(100))
    0.4373880352704239
    """
    return np.sqrt(2 * mse(y_pred, y))


def mae(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute error.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y : np.ndarray
        The true values.

    Returns
    -------
    float
        The mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> mae(np.random.rand(100), np.random.rand(100))
    0.3601070155483342
    """
    return np.mean(np.abs(y_pred - y))


def missclassification(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Missclassification error.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y : np.ndarray
        The true values.

    Returns
    -------
    float
        The missclassification error.
    """
    return np.mean(y_pred != y)


def accuracy(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Accuracy.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values.
    y : np.ndarray
        The true values.

    Returns
    -------
    float
        The accuracy.
    """
    return 1 - missclassification(y_pred, y)

def acc_F1(y_pred: np.ndarray, y: np.ndarray) -> float:
    """
    Return both accuracy and F1 score.
    
        Parameters
    ----------
    y_pred : np.ndarray
        The predicted values, either -1 or 1.
    y : np.ndarray
        The true values, either -1 or 1.

    Returns
    -------
    float
        The accuracy.
    float
        The F1 score.
    """
    return (accuracy(np.sign(y_pred), y), F1(np.sign(y_pred), y))


def F1(y_pred: np.ndarray, y: np.ndarray) -> float:
    """F1 score.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values, either -1 or 1.
    y : np.ndarray
        The true values, either -1 or 1.

    Returns
    -------
    float
        The F1 score.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> np.random.seed(1)
    >>> y_pred = np.random.choice([-1, 1], (100,))
    >>> y = np.random.choice([-1, 1], (100,))
    >>> F1(y_pred, y)
    0.49504950495049505
    """
    tp = np.sum((y_pred == 1) & (y == 1))
    fp = np.sum((y_pred == 1) & (y == -1))
    fn = np.sum((y_pred == -1) & (y == 1))
    F1_score = tp / (tp + 0.5 * (fp + fn))
    if np.isnan(F1_score):
        F1_score = 0.0
    return F1_score


def F1_continuous(y_pred: np.ndarray, y: np.ndarray) -> float:
    """F1 score when predictions are continuous values.

    Parameters
    ----------
    y_pred : np.ndarray
        The predicted values (continuous).
    y : np.ndarray
        The true values, either -1 or 1.

    Returns
    -------
    float
        The F1 score.
    """
    return F1(np.sign(y_pred), y)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

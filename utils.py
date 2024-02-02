import csv
import json
import os
import pickle
from typing import Generator

import numpy as np

from clustering import KMeans


def load_csv_data(
    data_path: str, sub_sample: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Load data from csv files and return numpy arrays.

    Parameters
    ----------
    data_path : str
        Datafolder path
    sub_sample : bool, optional
        If True the data will be subsempled, by default False

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list, list]
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def load_variables() -> dict:
    """Load variables from json file.

    Returns
    -------
    dict
        Dictionary with variables' informations. See variables.json for more details.
    """
    with open(os.path.join(os.path.dirname(__file__), "variables.json"), "r") as file:
        variables = json.load(file)
    return variables


def get_column_names(data_path: str) -> list:
    """Get column names from csv file.

    Parameters
    ----------
    data_path : str
        Path to data folder

    Returns
    -------
    list
        List of column names
    """
    header = np.loadtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", dtype=str, max_rows=1
    )
    return np.delete(header, 0).tolist()  # remove id column


def load_data(
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list, list]:
    """Load data from dataset_to_release folder path.

    Parameters
    ----------
    use_cache : bool, optional
        Wether to load the data from a cached file, or to create it if not present, by default True

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, list, list, list]
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        header (list): list of column names
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    data_path = os.path.join(os.path.dirname(__file__), "dataset_to_release")
    if use_cache:
        cache_data_path = os.path.join(os.path.dirname(__file__), "data_cache.pkl")
        if os.path.exists(cache_data_path):
            return pickle.load(
                open(
                    os.path.join(
                        cache_data_path,
                    ),
                    "rb",
                )
            )
        else:
            x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
            header = get_column_names(data_path)
            pickle.dump(
                (x_train, x_test, y_train, header, train_ids, test_ids),
                open(cache_data_path, "wb"),
            )
            return x_train, x_test, y_train, header, train_ids, test_ids
    else:
        x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
        header = get_column_names(data_path)
        return x_train, x_test, y_train, header, train_ids, test_ids


def batch_iter(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    num_batches: int = 1,
    shuffle: bool = True,
    seed: int = 1,
) -> Generator:
    """Generate a minibatch iterator for a dataset.

    Parameters
    ----------
    X : np.ndarray
        The training data
    y : np.ndarray
        The training labels
    batch_size : int
        The size of the mini-batches
    num_batches : int, optional
        The number of the mini-batches, by default 1
    shuffle : bool, optional
        Wether to shuffle the data, by default True
    seed : int, optional
        The random seed used to shuffle the data, by default 1

    Yields
    ------
    Generator
        A generator of mini-batches
    """
    data_size = y.shape[0]
    data_indices = np.arange(data_size)
    num_rep_data = np.ceil(num_batches / (data_size / batch_size)).astype(int)

    if shuffle:
        np.random.seed(seed)
        batch_data_indices = np.tile(np.random.permutation(data_indices), num_rep_data)
    else:
        batch_data_indices = np.tile(data_indices, num_rep_data)

    for batch_indices in np.array_split(
        batch_data_indices[: batch_size * num_batches], num_batches
    ):
        yield X[batch_indices], y[batch_indices]


def split_data(
    X: np.ndarray, y: np.ndarray, ratio: float, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing.

    Parameters
    ----------
    X : np.ndarray
        Data of shape (N, D), N is the number of samples, D is the number of features
    y : np.ndarray
        Labels of shape (N,)
    ratio : float
        Ratio of training data
    seed : int, optional
        Random number generator seed, by default 1

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    Examples
    --------
    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    N = X.shape[0]
    indices = np.random.permutation(N)
    N_tr = int(np.floor(N * ratio))
    x_tr, x_te = X[indices][:N_tr,], X[indices][N_tr:,]
    y_tr, y_te = y[indices][:N_tr,], y[indices][N_tr:,]
    return x_tr, x_te, y_tr, y_te


def create_csv_submission(ids: list[int], y_pred: np.ndarray, name: str) -> None:
    """This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Parameters
    ----------
    ids : list[int]
        indexes of the data
    y_pred : np.ndarray
        predictions on data correspondent to indices
    name : str
        name of the file to be created

    Raises
    ------
    ValueError
        If y_pred contains values different from -1 and 1
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def downsample_data(
    X: np.ndarray, y: np.ndarray, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Downsamples the data to have equal number of samples for each class.

    Parameters
    ----------
    X : np.ndarray
        The data to downsample
    y : np.ndarray
        The labels of the data
    seed : int, optional
        Random seed, by default 1

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The downsampled data and labels
    """
    # get indices of samples with label 1
    under_represented_idxs = np.argwhere(y == 1).squeeze()
    # get indices of samples with label -1
    over_represented_idxs = np.argwhere(y == -1).squeeze()
    # get number of samples to remove
    n_remove = over_represented_idxs.shape[0] - under_represented_idxs.shape[0]
    # get indices of samples to remove
    np.random.seed(seed)
    removed_idxs = np.random.choice(over_represented_idxs, size=n_remove, replace=False)
    # remove samples
    return np.delete(X, removed_idxs, axis=0), np.delete(y, removed_idxs, axis=0)


def downsample_centroid_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsamples the data to have equal number of samples for each class.
    """
    # get indices of samples with label 1
    under_represented_idxs = np.argwhere(y == 1).squeeze()
    # get indices of samples with label -1
    over_represented_idxs = np.argwhere(y == -1).squeeze()
    print(f"{under_represented_idxs.shape[0]=}")

    k = KMeans(under_represented_idxs.shape[0], 100)
    k.fit(X)
    new_data = k.centroids

    with open("centroid_data.pickle", "wb") as f:
        pickle.dump(new_data, f)

    X = np.delete(X, over_represented_idxs, axis=0)
    y = np.delete(y, over_represented_idxs, axis=0)

    return np.vstack((X, new_data)), np.hstack((y, -1 * np.ones(new_data.shape[0])))


def upsample_data(
    X: np.ndarray, y: np.ndarray, seed: int = 1, ratio: float = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Upsamples the data to have equal number of samples for each class.

    Parameters
    ----------
    X : np.ndarray
        The data to upsample
    y : np.ndarray
        The labels of the data
    seed : int, optional
        Random number seed, by default 1
    ratio : float, optional
        The ratio between event:non-event, by default 1

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        _description_
    """
    # get indices of samples with label 1
    under_represented_idxs = np.argwhere(y == 1).squeeze()
    # get indices of samples with label -1
    over_represented_idxs = np.argwhere(y == -1).squeeze()
    # How many times you can duplicate the under-represented samples
    n_duplicates = (np.floor(ratio * over_represented_idxs.shape[0])).astype(int) // (
        under_represented_idxs.shape[0]
    )

    expanded_under_represented_idxs = np.tile(under_represented_idxs, n_duplicates)

    remainder = (
        np.floor(ratio * over_represented_idxs.shape[0])
        % (expanded_under_represented_idxs.shape[0])
    ).astype(int)

    np.random.seed(seed)
    randomly_added_idxs = np.random.choice(
        under_represented_idxs, size=remainder, replace=False
    )
    expanded_under_represented_idxs = np.hstack(
        (expanded_under_represented_idxs, randomly_added_idxs)
    )

    return np.vstack(
        (X[over_represented_idxs], X[expanded_under_represented_idxs])
    ), np.hstack((y[over_represented_idxs], y[expanded_under_represented_idxs]))


if __name__ == "__main__":
    import timeit

    X_train, X_test, y_train, X_cols, _, _ = load_data()
    print(f"{X_train.shape=}")

    X_train, y_train = upsample_data(X_train, y_train)
    print(f"{X_train.shape=}")
    print(f"{y_train.shape=}")

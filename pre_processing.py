from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from utils import load_variables

VARIABLES = load_variables()


class PreProcessor(ABC):
    """Abstract class for pre-processing data"""

    def __init__(self) -> None:
        self.all_continuous_predictors = VARIABLES[
            "continuous_predictors"
        ]  # Discete & continuous predictors
        self.useless_predictors = VARIABLES["useless_predictors"]

    @abstractmethod
    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Transform the incoming data with it's column names and return the transformed data with it's the new column names

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names
        """
        pass

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Fit the processor on the incoming data and column names and return the transformed data with it's the new column names

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        return self.transform(data, col_names)

    def delete_columns(
        self,
        indices_to_remove: Union[list, np.ndarray],
        data: np.ndarray,
        col_names: list[str],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Delete specified variables in the data.

        Parameters:
        ------------
        indices_to_remove: list or np.ndarray
            The indices of the columns to remove.
        data: np.ndarray
            The data to modify.
        col_names: list[str]
            The column names of the data.

        Returns:
        A tuple containing the modified data and column names after removing specified columns.
        """
        data_clean = np.delete(data, indices_to_remove, axis=1)
        data_col_clean = np.delete(col_names, indices_to_remove)
        return data_clean, data_col_clean.tolist()

    def find_continuous_indexes(self, col_names: list[str]) -> list[int]:
        continuous_cols = list(
            set(col_names).intersection(set(self.all_continuous_predictors))
        )
        """
        Find the indexes of the continuous & discrete predictors in the data

        Parameters
        ----------
        col_names : list[str]
            The column names of the data

        Returns
        -------
        list[int]
            The indexes of the continuous predictors in the data
        """
        continuous_indexes = [
            col_names.index(continuous_col) for continuous_col in continuous_cols
        ]
        return continuous_indexes

    def find_categorical_indexes(self, col_names: list[str]) -> list[int]:
        categorical_cols = list(
            set(col_names).difference(set(self.all_continuous_predictors))
        )
        """
        Find the indexes of the categorical predictors in the data

        Parameters
        ----------
        col_names : list[str]
            The column names of the data

        Returns
        -------
        list[int]
            The indexes of the categorical predictors in the data
        """
        categorical_indexes = [
            col_names.index(categorical_col) for categorical_col in categorical_cols
        ]
        return categorical_indexes

    def find_useless_predictors_indexes(self, col_names: list[str]) -> list[int]:
        """
        Find the indexes of the useless predictors in the data

        Parameters
        ----------
        col_names : list[str]
            The column names of the data

        Returns
        -------
        list[int]
            The indexes of the useless predictors in the data
        """
        cols_to_remove = list(set(col_names).intersection(set(self.useless_predictors)))
        useless_indexes = [
            col_names.index(col_to_remove) for col_to_remove in cols_to_remove
        ]
        return useless_indexes


class OneHotEncoder(PreProcessor):
    def __init__(
        self,
        drop_last: bool = False,
    ) -> None:
        """
        One Hot Encoder

        Parameters
        ----------
        drop_last : bool, optional
            Whether to drop the last column of the one hot encoded data,
            as it is a linear combination of the rest of the columns, by default False
        """
        super().__init__()
        self.drop_last = drop_last

    def one_hot(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """One hot encode the data

        Parameters
        ----------
        data : np.ndarray
            The data to one hot encode
        col_names : list[str]
            The column names of the data

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The one hot encoded data and the new column names
        """
        temp = [np.unique(data_col, return_inverse=True) for data_col in data.T]
        enc_arr = np.array([temp_i[1] for temp_i in temp]).T
        new_col_names = [
            f"{col_name}__{u_class}"
            for temp_i, col_name in zip(temp, col_names)
            for u_class in temp_i[0]
        ]
        # Use an identity matrix to one hot encode the data
        ohe_data = np.concatenate(
            [np.eye(enc_col.max() + 1)[enc_col] for enc_col in enc_arr.T], axis=1
        )
        if self.drop_last:
            remove_indices = np.cumsum([temp_i[0].shape[0] for temp_i in temp]) - 1
            ohe_data = np.delete(ohe_data, remove_indices, axis=1)
            new_col_names = np.delete(np.array(new_col_names), remove_indices).tolist()
        return ohe_data, new_col_names

    def transform(
        self, data: np.ndarray, col_names=list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Transform the data by one hot encoding the categorical variables

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : _type_, optional
            The associated column names, by default list[str]

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        categorical_preds_idx = super().find_categorical_indexes(col_names)
        continuous_preds_mask = np.ones(data.shape[1], dtype=bool)
        continuous_preds_mask[categorical_preds_idx] = False

        categorical_ohe_data, categorical_ohe_col_names = self.one_hot(
            data[:, categorical_preds_idx],
            np.array(col_names)[categorical_preds_idx].tolist(),
        )

        return (
            np.concatenate(
                [data[:, continuous_preds_mask], categorical_ohe_data], axis=1
            ),
            categorical_ohe_col_names
            + np.array(col_names)[continuous_preds_mask].tolist(),
        )


class MissingValuesDropper(PreProcessor):
    def __init__(
        self,
        missing_percentage_cat: float = 0.95,
        missing_percentage_cont: float = 0.95,
    ) -> None:
        """
        Initialize the MissingValuesDropper

        Parameters
        ----------
        missing_percentage_cat : float, optional
            The percentage of missing values to consider a categorical variable as useless, by default 0.95
        missing_percentage_cont : float, optional
            The percentage of missing values to consider a continuous or discrete variable as useless, by default 0.95
        """

        super().__init__()
        self.missing_percentage_cat = missing_percentage_cat
        self.missing_percentage_cont = missing_percentage_cont

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Remove the Variables in self.indices

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        if not hasattr(self, "indices"):
            raise ValueError(
                "The indices of the columns to drop are not set, please call `fit_transform` first."
            )
        return super().delete_columns(self.indices, data, col_names)

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Find and Remove the Variables with too many missing values

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        cont_idxs = self.find_continuous_indexes(col_names)
        cat_idxs = self.find_categorical_indexes(col_names)

        indices_cat = np.array(cat_idxs)[
            np.where(
                np.sum(np.isnan(data[:, cat_idxs]), axis=0)
                >= data[:, cat_idxs].shape[0] * self.missing_percentage_cat
            )
        ]

        indices_cont = np.array(cont_idxs)[
            np.where(
                np.sum(np.isnan(data[:, cont_idxs]), axis=0)
                >= data[:, cont_idxs].shape[0] * self.missing_percentage_cont
            )
        ]

        self.indices = np.hstack((indices_cat, indices_cont))
        return self.transform(data, col_names)


class ConstantFeaturesRemover(PreProcessor):
    """
    Remove the constant features from the data
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """Remove the constant features from the data

        Parameters
        ----------
        eps : float, optional
            The standard deviation threshold to consider a feature as constant, by default 1e-6
        """
        super().__init__()
        self.eps = eps

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Remove the columns from self.cst_idxs in the data

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        if not hasattr(self, "cst_idxs"):
            raise ValueError(
                "The indices of the columns to drop are not set, please call `fit_transform` first."
            )
        return super().delete_columns(self.cst_idxs, data, col_names)

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Find and remove the constant features from the data

        Parameters
        ----------
        data : np.ndarray
            The data to remove the constant features from
        col_names : list[str]
            The column names of the data

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The data without the constant features and the new column names
        """
        self.cst_idxs = np.argwhere(np.nanstd(data, axis=0) <= self.eps)
        return self.transform(data, col_names)


class FeatureDropper(PreProcessor):
    def __init__(self) -> None:
        super().__init__()

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Remove the Variables in self.indices

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        if not hasattr(self, "useless_columns_indexes"):
            raise ValueError(
                "The indices of the columns to drop are not set, please call `fit_transform` first."
            )
        return super().delete_columns(self.useless_columns_indexes, data, col_names)

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Find and remove the useless features from the data

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        self.useless_columns_indexes = super().find_useless_predictors_indexes(
            col_names
        )
        return self.transform(data, col_names)


class ToNaNFiller(PreProcessor):
    def __init__(
        self,
        dtype: np.dtype = np.float64,
    ) -> None:
        """Fill the values of the columns to change with NaN

        Parameters
        ----------
        cols_to_change : list[str], optional
            The columns to change, by default None
        dtype : np.dtype, optional
            The dtype of the data, by default np.float32
        """
        super().__init__()
        self.dtype = dtype
        self.variables = VARIABLES  # Used for the signification of the answers

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Fill the values of the columns to change with NaN

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        self.cols_to_change = list(self.variables["signification"].keys())
        self.cols_to_change = set(self.cols_to_change).intersection(set(col_names))

        for desired_predcitor in self.cols_to_change:
            col_index = col_names.index(desired_predcitor)
            for answer, signification in self.variables["signification"][
                desired_predcitor
            ]["Values"].items():
                if (
                    signification
                    in [
                        "Don\u2019t know/Not sure",
                        "None",
                        "Refused",
                        "Refused/Missing",
                        "Refused----Go to next module",
                        "Don\u2019t know/Not sure----Go to next module",
                        "Don\u2019t know/Not Sure Or Refused/Missing",
                        "Don\u2019t know/Not Sure/Refused/Missing",
                        "Don\u2019t know/Not Sure, Refused or Missing",
                        "Don\u2019t know/Refused/Missing",
                        "Don\u2019t know/Not Sure/Refused",
                        "Don\u2019t know / Not sure",
                        "Don\u2019t know/Not Sure",
                        "Don\u2019t know/Not sure/Missing",
                        "Don\u2019t know/Not sure/Refused",
                        "Not applicable",
                        "Don’t know/Not sure",
                        "Don’t know/Not Sure",
                        "Don't Know/Not Sure",
                        "Don’t know/Not sure/Refused/Missing",
                        "Don’t know/Refused/Missing",
                        "Don’t know/Not sure/Missing",
                        "Don ́t know, refused or missing values",
                        "Don’t know/Not Sure/Refused/Missing",
                        "Never heard of \u2018\u2018A one C\u2019\u2019 test",
                        "Zero (none)",
                        # List of all answer to set to NaN
                    ]
                    and answer != "BLANK"
                ):
                    data[:, col_index] = np.where(
                        data[:, col_index] == self.dtype(answer),
                        np.nan,
                        data[:, col_index],
                    )
        return data, col_names


class Standardizer(PreProcessor):
    def __init__(self, with_nan: bool = True) -> None:
        """Standardize the data

        Parameters
        ----------
        with_nan : bool, optional
            Wether the data contains NaN, by default True
        """
        super().__init__()
        self.with_nan = with_nan

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Standardize the continuous and discrete variables with the mean and stds

        Parameters
        ----------
        data : np.ndarray
            The data to standardize
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The standardized data and the (unchanged) column names
        """
        if (
            not hasattr(self, "means")
            or not hasattr(self, "stds")
            or not hasattr(self, "continuous_indexes")
        ):
            raise ValueError(
                "The means and stds are not set, please call `fit_transform` first."
            )
        normalized_data = data[:]
        normalized_data[:, self.continuous_indexes] = (
            normalized_data[:, self.continuous_indexes] - self.means
        ) / self.stds
        return normalized_data, col_names

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Standardize the continuous and discrete variables

        Parameters
        ----------
        data : np.ndarray
            The data to standardize
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The standardized data and the (unchanged) column names
        """
        self.continuous_indexes = super().find_continuous_indexes(col_names)
        if self.with_nan:
            self.means = np.nanmean(data[:, self.continuous_indexes], axis=0)
            self.stds = np.nanstd(data[:, self.continuous_indexes], axis=0)
        else:
            self.means = np.mean(data[:, self.continuous_indexes], axis=0)
            self.stds = np.std(data[:, self.continuous_indexes], axis=0)
        return self.transform(data, col_names)


class PropPredictorsRemover(PreProcessor):
    """
    Remove the correlated predictors from the data
    """

    def __init__(self):
        pass

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Remove one of each correlated pairs of predictors from the data

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        if not hasattr(self, "corr_pred"):
            raise ValueError(
                "The indices of the columns to drop are not set, please call `fit_transform` first."
            )
        if self.corr_pred is None:
            return data, col_names
        else:
            return super().delete_columns(self.corr_pred, data, col_names)

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Find and remove the correlated predictors from the data

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        corr_matrix = np.corrcoef(data, rowvar=False)
        np.fill_diagonal(
            corr_matrix, 0
        )  # Set diagonal elements to 0 to exclude self-correlation
        corr_pairs = np.argwhere(np.isclose(corr_matrix, 1))
        if len(corr_pairs) == 0:
            self.corr_pred = None  # No perfectly correlated columns found
        else:
            self.corr_pred = np.unique(np.sort(corr_pairs), axis=0)[:, 0]
        return self.transform(data, col_names)


class FillImputer(PreProcessor):
    def __init__(self, strategy: str = "median") -> None:
        """Fill the NaN values of the data

        Parameters
        ----------
        strategy : str
            The method to fill the NaN values, either `mean` or `median`
        """
        super().__init__()
        self.strategy = strategy

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Fill the NaN values of the data previously computed values

        Parameters
        ----------
        data : np.ndarray
            The data to fill the NaN values of
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The data with the NaN values filled and the (unchanged) column names
        """
        if not hasattr(self, "fill_values") or not hasattr(self, "continuous_indexes"):
            raise ValueError(
                "The fill values are not set, please call `fit_transform` first."
            )
        filled_data = data[:]
        filled_data[:, self.continuous_indexes] -= self.fill_values
        filled_data[:, self.continuous_indexes] = np.where(
            np.isnan(filled_data[:, self.continuous_indexes]),
            0.0,
            filled_data[:, self.continuous_indexes],
        )
        filled_data[:, self.continuous_indexes] += self.fill_values

        if self.strategy == "distributions":
            for cat_idx, cat_distr in self.categorical_distributions.items():
                nans = np.isnan(data[:, cat_idx])
                filled_data[nans, cat_idx] = cat_distr(nans.sum())

        return filled_data, col_names

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Fill the NaN values of the data and compute the values to fill with

        Parameters
        ----------
        data : np.ndarray
            The data to fill the NaN values of
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The data with the NaN values filled and the (unchanged) column names

        Raises
        ------
        ValueError
            If the method to fill the NaN values is not implemented
        """
        self.continuous_indexes = super().find_continuous_indexes(col_names)
        if self.strategy == "median":
            self.fill_values = np.nanmedian(data[:, self.continuous_indexes], axis=0)
        elif self.strategy == "mean":
            self.fill_values = np.nanmean(data[:, self.continuous_indexes], axis=0)
        elif self.strategy == "distributions":
            self.fill_values = np.nanmedian(data[:, self.continuous_indexes], axis=0)

            self.categorical_indexes = self.find_categorical_indexes(col_names)
            self.categorical_distributions = {}
            for categorical_index in self.categorical_indexes:
                self.categorical_distributions.update(
                    {
                        categorical_index: self.find_categorical_distribution(
                            data, categorical_index
                        )
                    }
                )

        else:
            raise ValueError(
                f"Method to fill nan values `{self.strategy}` is not implemented."
            )
        return self.transform(data, col_names)

    def find_categorical_distribution(self, X: np.ndarray, col_idx: int):
        """
        Find the distribution of the categorical variable

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, where N is the number of samples and D is the number of features.
        col_idx : int
            The index of the categorical variable

        Returns
        -------
        function
            The distribution of the categorical variable
        """
        nans = np.isnan(X[:, col_idx])
        labels, counts = np.unique(X[nans == False, col_idx], return_counts=True)
        distr = lambda num_points: np.random.choice(
            labels, (num_points,), p=counts / np.sum(counts)
        )
        return distr


class CategoricalDropper(PreProcessor):
    """
    Remove the categorical variables from the data
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Remove the categorical variables from the data

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        if not hasattr(self, "categorical_indexes"):
            raise ValueError(
                "The indices of the columns to drop are not set, please call `fit_transform` first."
            )
        return super().delete_columns(self.categorical_indexes, data, col_names)

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Find and remove the categorical variables from the data

        Parameters
        ----------
        data : np.ndarray
            The data to transform
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        self.categorical_indexes = super().find_categorical_indexes(col_names)
        return self.transform(data, col_names)


class BiasAdder(PreProcessor):
    """
    Add a bias column to the data
    """

    def __init__(self) -> None:
        super().__init__()

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Add a bias column to the data

        Parameters
        ----------
        data : np.ndarray
            Feature matrix, where N is the number of samples and D is the number of features.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        return (
            np.concatenate([np.ones((data.shape[0], 1)), data], axis=1),
            ["bias"] + col_names,
        )

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        return self.transform(data, col_names)


class Pipeline(PreProcessor):
    """
    Implementation of a pipeline of PreProcessors
    """

    def __init__(self, steps: list) -> None:
        self.functions = []
        for step, arg in steps:
            # initialize
            if arg != None:
                self.functions.append(step(**arg))
            else:
                self.functions.append(step())

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Transform the data with all steps of the pipeline

        Parameters
        ----------
        data : np.ndarray
            Feature matrix, where N is the number of samples and D is the number of features.
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        for function in self.functions:
            print(f"Processing step: {function.__class__.__name__}")
            data, col_names = function.transform(data, col_names)
        return data, col_names

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Fit and transform the data with all steps of the pipeline

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names

        Raises
        ------
        """
        for function in self.functions:
            print(f"Processing step: {function.__class__.__name__}")
            data, col_names = function.fit_transform(data, col_names)
        return data, col_names


class PolynomialFeatures(PreProcessor):
    """
    Add polynomial features to the data

    Parameters
    ----------
    degree : int
        The degree of the polynomial features to add

    Returns
    -------
    tuple[np.ndarray, list[str]]
        The transformed data and the new column names
    """

    def __init__(self, degree: int) -> None:
        super().__init__()
        self.degree = degree

    def transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Add polynomial features to the data

        Parameters
        ----------
        data : np.ndarray
            Feature matrix, where N is the number of samples and D is the number of features.
        col_names : list[str]
            The associated column names

        Returns
        -------
        tuple[np.ndarray, list[str]]
            The transformed data and the new column names
        """
        if not hasattr(self, "continuous_cols") or not hasattr(
            self, "categorical_cols"
        ):
            raise ValueError(
                "The indices of the columns use are not set, please call `fit_transform` first."
            )
        polynomial_continous_cols = self.build_poly(data[:, self.continuous_cols])
        polynomial_continuous_col_names = self.change_col_names(
            np.array(col_names)[self.continuous_cols].tolist()
        )
        return (
            np.concatenate(
                [data[:, self.categorical_cols], polynomial_continous_cols], axis=1
            ),
            np.array(col_names)[self.categorical_cols].tolist()
            + polynomial_continuous_col_names,
        )

    def fit_transform(
        self, data: np.ndarray, col_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """
        Find the continuous+discrete and categorical columns and add polynomial features to the data

        Parameters
        ----------
        data : np.ndarray
            Feature matrix, where N is the number of samples and D is the number of features.
        col_names : list[str]
            The associated column names

        """
        self.continuous_cols = super().find_continuous_indexes(col_names)
        self.categorical_cols = super().find_categorical_indexes(col_names)
        return self.transform(data, col_names)

    def change_col_names(self, col_names: list[str]) -> list[str]:
        """
        Change the column names to add the degree of the polynomial features

        Parameters
        ----------
        col_names : list[str]
            The associated column names

        Returns
        -------
        list[str]
            The new column names
        """
        return col_names + [
            col_name + "__^" + str(i)
            for i in range(2, self.degree + 1)
            for col_name in col_names
        ]

    def build_poly(self, X: np.ndarray) -> np.ndarray:
        """
        Build the polynomial features

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, where N is the number of samples and D is the number of features.

        Returns
        -------
        np.ndarray
            The polynomial features
        """
        return np.concatenate([X**j for j in range(1, self.degree + 1)], axis=1)


if __name__ == "__main__":
    # artificial_data = np.array(
    #     [
    #         [0, 1, 2, 0, 0],
    #         [0, 1, 2, 3, 4],
    #         [0, 0, 0, 1, 1],
    #     ]
    # ).T
    # artificial_data_col_names = ["col1", "col2", "col3"]
    # print((artificial_data, artificial_data_col_names))
    # print(OHE(drop_last=True).fit_transform(artificial_data, artificial_data_col_names))

    import timeit

    from utils import load_data

    X, X_test, y, X_cols, _, _ = load_data()

    X = np.concatenate([X, X_test], axis=0)

    # X, y = X[:10], y[:10]

    print(f"{X.shape=}")

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
            (PolynomialFeatures, {"degree": 2}),
        ],
    )

    start = timeit.default_timer()
    X, X_cols = preprocessor_pipe.fit_transform(X, X_cols)
    end = timeit.default_timer()
    print(f"{end-start=}")

    print(f"{X.shape=}")
    print(f"{len(X_cols)=}")

    # X, X_test, y, X_cols, _, _ = load_data()
    # # X[:2100], y[:2100]

    # preprocessor_pipe = Pipeline(
    #     steps=[
    #         (FeatureDropper, None),
    #         (ToNaNFiller, None),
    #         (MissingValuesDropper, None),
    #         (ConstantFeaturesRemover, None),
    #         (FillImputer, {"strategy": "median"}),
    #         (PropPredictorsRemover, None),
    #         (Standardizer, {"with_nan": False}),
    #         (OneHotEncoder, {"drop_last": True}),
    #         # (PolynomialFeatures, {"degree": 2}),
    #     ],
    # )
    # print(f"{X.shape=}")
    # start_preprocessing = timeit.default_timer()
    # X, X_train_cols = preprocessor_pipe.fit_transform(X, deepcopy(X_cols))
    # end_preprocessing = timeit.default_timer()
    # print(f"{X.shape=}")
    # print(f"{end_preprocessing-start_preprocessing=}")

    # X_train, y_train = downsample_data(X, y)
    # y_train = np.where(y_train == -1, 0, y_train)

    # X_test, X_test_cols = preprocessor_pipe.transform(X_test, deepcopy(X_cols))
    # print(f"{X_test.shape=}")
    # print(f"{X_train_cols==X_test_cols=}")
    # print(f"{np.isnan(X_test).sum()=}")

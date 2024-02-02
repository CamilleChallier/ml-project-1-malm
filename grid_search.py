from copy import deepcopy
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from cross_validation import CrossValidation, StratifiedUpsampledCV
from utils import split_data


class GridSearch:
    def __init__(
        self,
        model_class,
        parameter_grid: dict,
        metric,
        constant_params: dict = {},
        eval_criteria: str = "max",
        k_folds: int = None,
        ratio: float = 0.8,
        seed: int = 1,
        verbose: bool = False,
    ) -> None:
        """Grid search class

        Parameters
        ----------
        model_class
            The model class to use
        parameter_grid : dict
            A dictionary of the form {hyperparameter_name : [values_to_test]}, e.g. {"gamma" : [0.1, 0.01, 0.001]}
        metric : function
            A function that takes as input y_pred and y_true and returns a scalar score
        constant_params : dict, optional
            A dictionary of the form {hyperparameter_name : value}, e.g. {"max_iters" : 100}, by default {}
        eval_criteria : str, optional
            The evaluation criteria, either "max" or "min", by default "max"
        k_folds : int, optional
            The number of folds to use for cross validation, if None cross validation is not used and holdout method is used, by default None
        ratio : float, optional
            The ratio of data to use for training with holdout method, only useful if k_folds=None, by default 0.8
        seed : int, optional
            The random seed number, by default 1
        verbose : bool, optional
            Wether to print the results of the grid search, by default False
        """
        self.parameter_grid = parameter_grid
        self.constant_params = constant_params
        self.parameter_combinations = (
            self._create_combinations()
        )  # Create all the combinations of hyperparameters to test
        self.model_class = model_class
        self.metric = metric
        if eval_criteria == "max":
            self.eval_criteria = np.argmax
        elif eval_criteria == "min":
            self.eval_criteria = np.argmin
        else:
            raise ValueError("The evaluation criteria should be either `max` or `min`.")
        self.k_folds = k_folds
        self.ratio = ratio
        self.seed = seed
        self.verbose = verbose
        self.scores = None

    def _create_combinations(self) -> list:
        """Create all the combinations of hyperparameters to test"""
        return [
            {key: param for key, param in zip(self.parameter_grid.keys(), params)}
            for params in product(*self.parameter_grid.values())
        ]

    def _fit_CV(
        self, params: dict, X: np.ndarray, y: np.ndarray
    ) -> Union[tuple[float, float], float]:
        """Fit the model with cross validation"""
        results = CrossValidation(
            self.model_class(**params, **self.constant_params),
            k_folds=self.k_folds,
            metric=self.metric,
            seed=self.seed,
        ).evaluate(X, y)
        if self.verbose:
            print(
                "-" * 50,
                "\n",
                f"For the parameters {params} the results of CV are:\n{results}",
                "\n",
                "-" * 50,
            )
        return results

    def _fit_holdout(
        self,
        params: dict,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> Union[tuple[float, float], float]:
        """Fit the model with holdout method"""
        model = self.model_class(**params, **self.constant_params)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        results = (self.metric(y_test_pred, y_test), self.metric(y_train_pred, y_train))
        if self.verbose:
            print(
                "-" * 50,
                "\n",
                f"For the parameters {params} the results of holdout method are:\n{results}",
                "\n",
                "-" * 50,
            )
        return results

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model with the different hyperparameter combinations

        Parameters
        ----------
        X : np.ndarray
            The data
        y : np.ndarray
            The labels
        """
        if self.k_folds is not None:
            # If k_folds is not None, use cross validation
            scores = [
                self._fit_CV(params, X, y) for params in self.parameter_combinations
            ]
            self.scores = {
                "test_scores_averaged": [
                    score["test_scores_averaged"] for score in scores
                ],
                "test_scores_per_fold": [
                    score["test_scores_per_fold"] for score in scores
                ],
                "train_scores_averaged": [
                    score["train_scores_averaged"] for score in scores
                ],
                "train_scores_per_fold": [
                    score["train_scores_per_fold"] for score in scores
                ],
            }
        else:
            # Use the holdout method where we split the data into train and test
            X_train, X_test, y_train, y_test = split_data(
                X, y, ratio=self.ratio, seed=self.seed
            )
            scores = np.array(
                [
                    self._fit_holdout(params, X_train, X_test, y_train, y_test)
                    for params in self.parameter_combinations
                ]
            )  # Dimension: (num_param_combinations, 2) where the second dimension is (test_score, train_score)
            self.scores = {
                "test_scores": scores[:, 0],
                "train_scores": scores[:, 1],
            }

        return self

    def get_best_parameters(
        self,
    ) -> tuple[dict, dict]:
        """Return the best parameters and the corresponding score

        Returns
        -------
        tuple[dict, dict]
            The best parameters and all of the scores
        """
        if self.k_folds is None:
            best_test_score_idx = self.eval_criteria(self.scores["test_scores"])
        else:
            best_test_score_idx = self.eval_criteria(
                self.scores["test_scores_averaged"]
            )

        return (
            self.parameter_combinations[best_test_score_idx] | self.constant_params,
            self.scores,
        )

    def plot_results(
        self,
        parameter_scale: dict[str:str] = None,
        savename: str = None,
        show: bool = False,
        cmap: str = "plasma",
        figsize=(10, 7),
    ) -> None:
        """Plot the results of the grid search

        Parameters
        ----------
        parameter_scale : dict, optional
            A dictionary of the form {hyperparameter_name : scale}, where scale is either "linear" or "log", by default None
        savename : str, optional
            The name of the file to save the plot, by default None
        show : bool, optional
            Wether to show the plot, by default False
        cmap : str, optional
            The name of the colormap to use, by default "plasma"
        figsize : tuple, optional
            The size of the figure, by default (10, 7)
        """
        # Get arrays that are handy to plot
        if self.k_folds is None:
            test_scores = self.scores["test_scores"]
        else:
            test_scores = self.scores["test_scores_averaged"]

        parameter_scale_ = {
            param_name: "linear" for param_name in self.parameter_grid.keys()
        }
        if parameter_scale is not None:
            parameter_scale_.update(parameter_scale)

        num_hyper_params = len(
            list(self.parameter_grid.keys())
        )  # Number of hyperparameters
        fig, axes = plt.subplots(num_hyper_params, num_hyper_params, figsize=figsize)

        if num_hyper_params == 1:
            # So you can index the axes
            axes = np.array([axes]).reshape(1, 1)

        plotting_values = np.array(list(product(*self.parameter_grid.values())))
        # Create a dictionary of the form {hyperparameter_name : values_tested}
        plotting_dict = {
            param_name: values_tested
            for param_name, values_tested in zip(
                self.parameter_grid.keys(), plotting_values.T
            )
        }
        if num_hyper_params > 1:
            self._add_colorbar_to_plot(
                test_scores, cmap, axes[:-1, num_hyper_params - 1]
            )
        for i, key_i in enumerate(self.parameter_grid.keys()):
            for j, key_j in enumerate(self.parameter_grid.keys()):
                if i == j:
                    # If you are on the diagonal plot the parameter vs the score
                    kwargs = dict(x=plotting_dict[key_i], y=test_scores)
                    if num_hyper_params > 1:
                        kwargs.update(dict(c=test_scores, cmap=cmap))
                    axes[i, j].scatter(**kwargs)
                    axes[i, j].set_ylabel("score")
                    axes[i, j].set_xscale(parameter_scale_[key_i])
                    if i == num_hyper_params - 1:
                        # If you are on the last row add the parameter name on x-axis
                        axes[i, j].set_xlabel(key_i)
                    else:
                        axes[i, j].set_xticklabels([])
                    if num_hyper_params > 1:
                        axes[i, j].yaxis.set_ticks_position("right")
                        axes[i, j].yaxis.set_label_position("right")
                else:
                    if i < j:
                        # Hide the upper diagonal
                        axes[i, j].set_visible(False)
                    else:
                        kwargs = dict(x=plotting_dict[key_j], y=plotting_dict[key_i])
                        if num_hyper_params != 1:
                            kwargs.update(dict(c=test_scores, cmap=cmap))
                        axes[i, j].scatter(**kwargs)
                        axes[i, j].set_yscale(parameter_scale_[key_i])
                        axes[i, j].set_xscale(parameter_scale_[key_j])
                        if i == num_hyper_params - 1:
                            # If on last row, set the xlabel
                            axes[i, j].set_xlabel(key_j)
                        else:
                            # Remove the xtick labels if not on last row
                            axes[i, j].set_xticklabels([])
                        if j == 0:
                            # If on first columns set the ylabel
                            axes[i, j].set_ylabel(key_i)
                        else:
                            # Remove the ytick labels if not on first column
                            axes[i, j].set_yticklabels([])

        if savename is not None:
            if not savename.endswith(".png"):
                savename = f"{savename}.png"
            plt.savefig(savename)
        if show:
            plt.show()
        plt.close()

    def _add_colorbar_to_plot(self, test_scores: np.ndarray, cmap: str, axes) -> None:
        """Add a colorbar to the plot"""
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(test_scores)
        cbar = plt.colorbar(sm, ax=axes)
        cbar.set_label("score")


class StrtifiedUpsampledGridSearchCV(GridSearch):
    def __init__(
        self,
        model_class,
        parameter_grid: dict,
        metric,
        k_folds: int,
        constant_params: dict = {},
        eval_criteria: str = "max",
        seed: int = 1,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            model_class=model_class,
            parameter_grid=parameter_grid,
            metric=metric,
            constant_params=constant_params,
            eval_criteria=eval_criteria,
            k_folds=k_folds,
            seed=seed,
            verbose=verbose,
        )
        self.strat_up_CV = StratifiedUpsampledCV(
            n_splits=self.k_folds, random_state=self.seed
        )

    def _fit_CV(
        self, params: dict, X: np.ndarray, y: np.ndarray
    ) -> Union[tuple[float, float], float]:
        """Fit the model with stratified upsampled cross validation"""
        results = {
            "test_scores_per_fold": [],
            "train_scores_per_fold": [],
        }
        for train_indexes, test_indexes in self.strat_up_CV.split(X, y):
            model = self.model_class(**params, **self.constant_params).fit(
                X[train_indexes], y[train_indexes]
            )
            results["test_scores_per_fold"].append(
                self.metric(model.predict(X[test_indexes]), y[test_indexes])
            )
            results["train_scores_per_fold"].append(
                self.metric(model.predict(X[train_indexes]), y[train_indexes])
            )
        results["test_scores_averaged"] = np.mean(results["test_scores_per_fold"])
        results["train_scores_averaged"] = np.mean(results["train_scores_per_fold"])
        if self.verbose:
            print(
                "-" * 50,
                "\n",
                f"For the parameters {params} the results of CV are:\n{results}",
                "\n",
                "-" * 50,
            )
        return results


if __name__ == "__main__":
    import timeit

    from base_regressors import LogisticRegressor
    from metrics import F1, F1_continuous, accuracy
    from pre_processing import *
    from utils import load_data

    X, _, y, X_cols, _, _ = load_data()

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

    print(f"{X.shape=}")
    start_preprocessing = timeit.default_timer()
    X, X_cols = preprocessor_pipe.fit_transform(X, X_cols)
    end_preprocessing = timeit.default_timer()
    print(f"{X.shape=}")
    print(f"{end_preprocessing-start_preprocessing=}")

    print(f"{np.isnan(X).sum()=}")

    grid_search = StrtifiedUpsampledGridSearchCV(
        LogisticRegressor,
        {
            # "lambda_": np.logspace(-5, 0, 3),
            "max_iters": [300],
            "gamma": [0.324742],
        },
        accuracy,
        k_folds=3,
        seed=60,
        verbose=True,
    )
    grid_search.fit(X, y)
    print(f"{grid_search.get_best_parameters()=}")
    grid_search.plot_results(
        # parameter_scale={
        #     # "gamma": "log",
        #     # "lambda_": "log",
        # },
        savename="ttteeesssttt.png",
        cmap="plasma",
    )

    print(f"{grid_search.parameter_combinations=}")

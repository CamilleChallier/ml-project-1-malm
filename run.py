# run our best model
import timeit

from base_regressors import *
from base_regressors import LogisticRegressor
from cross_validation import CrossValidation
from enhanced_regressors import *
from enhanced_regressors import ImbalancedLogisticRegressor, LLRENSRegressor
from metrics import acc_F1
from pre_processing import *
from utils import create_csv_submission, load_data, upsample_data

if __name__ == "__main__":
    X_train, X_test, y_train, X_cols, train_idxs, test_idxs = load_data()  # load data
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
    )  # all preprocessing steps
    print(f"{X_train.shape=}")
    start_preprocessing = timeit.default_timer()
    X_train, X_train_cols = preprocessor_pipe.fit_transform(
        X_train, deepcopy(X_cols)
    )  # preprocess data
    end_preprocessing = timeit.default_timer()
    print(f"{X_train.shape=}")
    print(f"{end_preprocessing-start_preprocessing=}")

    X_test, X_test_cols = preprocessor_pipe.transform(
        X_test, deepcopy(X_cols)
    )  # transform test data
    assert X_train_cols == X_test_cols

    # upsample data
    X_train_up, y_train_up = upsample_data(X_train, y_train)

    # define model
    model = LogisticRegressor(
        gamma=0.324742, max_iters=300, learning="gradient_descent"
    ).fit(
        X_train_up, y_train_up
    )  # fit model

    y_test_pred = model.predict(X_test)  # make predictions
    create_csv_submission(
        test_idxs, y_test_pred, "ULR_best.csv"
    )  # create AI crowd submission file

# MALM
### MA(ster's) M(achine) L(earning) (course)
![image](https://github.com/epfml/ml-project-1-malm/assets/81494719/f3f856be-0d71-47f0-a486-04f81450306a)

## *Machine Learning Course Project 1*

## Aim

The aim of the project is to gain practical experience by applying the principles covered in our lectures and practiced in our labs to a real-world dataset. Our journey will involve conducting exploratory data analysis to comprehend the dataset and extract valuable insights, and applying machine learning techniques to generate predictions, all of which will be documented in our findings.

## :handshake: Contributors

Gonçalo Braga, Wesley Monteith, Camille Challier


## Project description
This project deals with the classification of Coronary Heart Disease positive individuals based on the Behavioral Risk Factor Surveillance System (BRFSS) data set.

## Environment:

To make a new `malm` conda environment please execute the following line (not sure it works for windows powershell though):
Linux:
```bash
sh setup.sh
```
Windows:
```bash
wsl sh setup.sh
```

## Codestyle
To simplify transition of users from Scikit Learn, a similar API in our coding style was chosen.

## Repository structure

| Path | Description
| :--- | :----------
| ml-project-1-malm | Repository root folder of the ML course 1st project.
| &boxvr;&nbsp; Figures.ipynb | Notebook to re-create the figures of the paper.
| &boxvr;&nbsp; README.md | Readme file for the project, providing an overview and instructions.
| &boxvr;&nbsp; base_regressors.py | Python script containing Least Squares, Ridge, Gradient Descent and Logistic Regressor methods.
| &boxvr;&nbsp; clustering.py | Implementation of Kmeans algorithm.
| &boxvr;&nbsp; cross_validation.py | Python script containing CrossValidation implementation.
| &boxvr;&nbsp; enhanced_regressors.py | Python script containing other regressor methods : Lasso, ElasticNet, Polynomial, Imbalanced Logistic and LLRENS Regressor.
| &boxvr;&nbsp; grid_search.py | Implementation of grid search algorithm.
| &boxvr;&nbsp; implementations.py | Implementation of the functions see in class.
| &boxvr;&nbsp; metrics.py | Definition of all metrics functions used in this project : mse, rmse, mae, missclassification rate, F1, accuracy.
| &boxvr;&nbsp; pre_processing.py | Python script containing all steps of pre-processing.
| &boxvr;&nbsp; run.py | Run our best model, describe the pre processing, cross validation and prediction steps
| &boxvr;&nbsp; setup.sh | Environment definition.
| &boxvr;&nbsp; utils.py | Contains useful functions.
| &boxvr;&nbsp; variables.json | Python script containing the name of all variables with their description, their types and utilities.


## Hyperparameters for the best runs of each model

| Method | Hyperparameters
| :------| :---------------
| Ridge | $\lambda =  1.27 \cdot 10^{-6}$
| Lasso | $\lambda$ = 10, $\gamma$ = 0.0001, I = 100
| Elastic Net | $\gamma$ = 0.032, $\lambda$ = 10, $l1_{ratio}$ = 10
| GD | $\gamma$ = 0.1, I = 180, $\mathcal{L}$ = MSE
| Mini-batch GD | $\gamma$ = 0.03, I = 80, $\mathcal{L}$ = MAE, BS = 64
| SGD | $\gamma$ = 0.0125, I = 300, $\mathcal{L}$ = MAE
| Polynomial Ridge | $\lambda = 1 \cdot 10^{-5}$ , degree = 3
| LR | $\gamma$ = 0.3, I = 100
| Upsampled LR | $\gamma$ = 0.3, I = 100
| Downsampled LR | $\gamma$ = 0.2, I = 500
| L2 LR | $\lambda = 3.54 \cdot 10^{-6}$, $\gamma$ = 1.55, I = 390
| Imbalanced LR | $\gamma$ = 0.4, $\alpha$ = 0.06
| LLRENS | $\gamma_{i}$ = \{0.001, 4.341e-06, 10, 1e-06, 1.144e-06, 1e-06, 0.015, 1.531e-06, 1e-06, 7.536e-05\}  <br /> $\lambda_{i}$ = \{0.128, 1.050e-06, 2.419, 1e-06, 1e-06, 3.148e-05, 9.347, 9.919, 10, 1.139e-05\} <br /> $I_{i}$ = \{450, 156, 195, 70, 10, 10, 491, 37, 37, 500\}

Note: All unspecified parameters are the default ones in our code.
KEY: I - max number of iterations, BS - batch size


## Regarding the LLRENS model in `enhanced_regressors.py`

- When the number of data points in the cluster was bigger then the event class data points, we bootstrapped the event data points (resampling with replacement).

- For prediction the authors suggested to use a system of weight voting based on the performance of the LLR on the testing data, but we observed that predicting an event when 30\% of LLRs agreed, yielded the best F1 score.

- We chose k=10.

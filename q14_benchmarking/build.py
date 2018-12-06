# %load q14_benchmarking/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression
from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor


from greyatomlib.multivariate_regression_project.q08_linear_model.build import linear_model
from greyatomlib.multivariate_regression_project.q12_feature_selection.build import feature_selection

from greyatomlib.multivariate_regression_project.q09_advanced_model_q01_lasso.build import lasso
from greyatomlib.multivariate_regression_project.q09_advanced_model_q02_ridge.build import ridge

from greyatomlib.multivariate_regression_project.q13_plot_residuals.build import plot_residuals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(7)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test = split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


# Write your code below
def create_stats(x_train, x_test, y_train, y_test):
    lasso_modl, lasso_y_pred, lasso_stat = lasso(x_train, x_test, y_train, y_test,0.1)
    ridge_modl, ridge_y_pred, ridge_stat = ridge(x_train, x_test, y_train, y_test,0.1)
    features = feature_selection(x_train, y_train, k=50)
    
    x_train_ft = x_train[features].copy()
    x_test_ft = x_test[features].copy()
    
    lasso_modl_ft, lasso_y_pred_ft, lasso_stat_ft = lasso(x_train_ft, x_test_ft, y_train, y_test,0.1)
    ridge_modl_ft, ridge_y_pred_ft, ridge_stat_ft = ridge(x_train_ft, x_test_ft, y_train, y_test,0.1)
    features = feature_selection(x_train_ft, y_train, k=50)
    
    complete_stats = pd.concat([lasso_stat,lasso_stat_ft,ridge_stat,ridge_stat_ft])
    
    return complete_stats

create_stats(x_train, x_test, y_train, y_test)



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

def create_stats(x_train, x_test, y_train, y_test):
    
    features = feature_selection(x_train, y_train, 50)
    x_train_transform = x_train[features]
    x_test_transform = x_test[features]
    
    _, _, stats_lasso = lasso(x_train, x_test, y_train, y_test)
    _, _, stats_lasso_ft = lasso(x_train_transform, x_test_transform, y_train, y_test)
    
    _, _, stats_ridge = ridge(x_train, x_test, y_train, y_test)
    _, _, stats_ridge_ft = ridge(x_train_transform, x_test_transform, y_train, y_test)
    
    complete_stats = pd.concat([stats_lasso,stats_lasso_ft,stats_ridge,stats_ridge_ft])
    return complete_stats





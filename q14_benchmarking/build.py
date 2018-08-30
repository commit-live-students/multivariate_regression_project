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
    
    l1,y_pred_l1,lasso_stats = lasso(x_train, x_test, y_train, y_test, alpha=0.1)
    l2,y_pred_l2,ridge_stats = ridge(x_train, x_test, y_train, y_test, alpha=0.1)
    
    features= feature_selection(x_train, y_train, k=50)
    x_trainft=x_train[features].copy()
    x_testft=x_test[features].copy()
    l1ft,y_pred_l1ft,lasso_statsft = lasso(x_trainft, x_testft, y_train, y_test, alpha=0.1)
    l2ft,y_pred_l2ft,ridge_statsft = ridge(x_trainft, x_testft, y_train, y_test, alpha=0.1)
    complete_stats = pd.concat([lasso_stats,lasso_statsft,ridge_stats,ridge_statsft])
    return complete_stats
    
    



#complete_stats = create_stats(x_train, x_test, y_train, y_test)

#complete_stats.shape[0]*complete_stats.shape[1]


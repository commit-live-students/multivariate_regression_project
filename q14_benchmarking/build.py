from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode


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

x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


def create_stats(x_train, x_test, y_train, y_test,enc = "labelencoder"):
    lin_model,y_pred,stats_lin = linear_model(x_train, x_test, y_train, y_test,"linear baseline "+enc)    
    # plot_residuals(y_test,y_pred,"images/linear baseline "+enc)
    lasso_model,y_pred,stats_lasso = lasso(x_train, x_test, y_train, y_test,"lasso baseline "+enc,alpha=0.1)
    # plot_residuals(y_test,y_pred,"images/lasso baseline "+enc)

    ridge_model,y_pred,stats_ridge = ridge(x_train, x_test, y_train, y_test,"ridge baseline "+enc,alpha=0.1)
    # plot_residuals(y_test,y_pred,"images/ridge baseline "+enc)

    feature_list = feature_selection(x_train,y_train,50)
    
    x_train = x_train[feature_list]
    x_test = x_test[feature_list]
    
    lin_model,y_pred,stats_lin_ft = linear_model(x_train, x_test, y_train, y_test,"linear ft "+enc)
    # plot_residuals(y_test,y_pred,"images/linear ft "+enc)

    lasso_model,y_pred,stats_lasso_ft = lasso(x_train, x_test, y_train, y_test,"lasso ft "+enc,alpha=0.1)
    # plot_residuals(y_test,y_pred,"images/lasso ft "+enc)

    ridge_model,y_pred,stats_ridge_ft = ridge(x_train, x_test, y_train, y_test,"ridge ft "+enc,alpha=0.1)
    # plot_residuals(y_test,y_pred,"images/ridge ft "+enc)

    complete_stats = pd.concat([stats_lin,stats_lin_ft,stats_lasso,stats_lasso_ft,stats_ridge,stats_ridge_ft])
    return complete_stats





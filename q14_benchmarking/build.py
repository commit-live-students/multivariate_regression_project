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
    # lin_model, y_pred_lin, stats_lin = linear_model(x_train, x_test, y_train, y_test)
    # plot_residuals(y_test,y_pred,"images/linear baseline "+enc)
    lasso_model, y_pred_lasso, stats_lasso = lasso(x_train, x_test, y_train, y_test, alpha=0.1)
    # plot_residuals(y_test,y_pred,"images/lasso baseline "+enc)

    ridge_model, y_pred_ridge, stats_ridge = ridge(x_train, x_test, y_train, y_test, alpha=0.1)
    # plot_residuals(y_test,y_pred,"images/ridge baseline "+enc)

    feature_list = feature_selection(x_train, y_train, 50)

    x_train = x_train[feature_list]
    x_test = x_test[feature_list]

    # lin_model_new, y_pred_lin_new, stats_lin_ft_new = linear_model(x_train, x_test, y_train, y_test)
    # plot_residuals(y_test,y_pred,"images/linear ft "+enc)

    lasso_model_new, y_pred_lasso_new, stats_lasso_ft_new = lasso(x_train, x_test, y_train, y_test, alpha=0.1)
    # plot_residuals(y_test,y_pred,"images/lasso ft "+enc)

    ridge_model_new, y_pred_ridge_new, stats_ridge_ft_new = ridge(x_train, x_test, y_train, y_test, alpha=0.1)
    # plot_residuals(y_test,y_pred,"images/ridge ft "+enc)

    complete_stats = pd.concat([stats_lasso, stats_lasso_ft_new, stats_ridge, stats_ridge_ft_new])
    return complete_stats

a = create_stats(x_train, x_test, y_train, y_test)
print(a.shape)




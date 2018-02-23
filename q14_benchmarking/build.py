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


def create_stats(X_train, X_test, y_train, y_test,enc = "labelencoder"):
    a, b,lm_score=linear_model(X_train, X_test, y_train, y_test,'')
    c, d,lasso_score=lasso(X_train, X_test, y_train, y_test)
    e, f,ridge_score=ridge(X_train, X_test, y_train, y_test)
    best_features = feature_selection(X_train,y_train, k=50)
    a, b,lm_score_bf=linear_model(X_train[best_features], X_test[best_features], y_train, y_test,'')
    c, d,lasso_score_bf=lasso(X_train[best_features], X_test[best_features], y_train, y_test)
    e, f,ridge_score_bf=ridge(X_train[best_features], X_test[best_features], y_train, y_test)

    complete_stats = pd.concat([lm_score,lasso_score,ridge_score,lm_score_bf,lasso_score_bf,ridge_score_bf],ignore_index=True)
    del complete_stats['rmse']
    return complete_stats#lm_score,lasso_score,ridge_score,lm_score_bf,lasso_score_bf,ridge_score_bf

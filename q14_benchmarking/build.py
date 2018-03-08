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

#Call basic model (Linear, Lasso and Ridge)
    lm_model, lm_y_pred,lm_stats = linear_model(x_train, x_test, y_train, y_test, 0.01)
    la_model, la_y_pred,la_stats = lasso(x_train, x_test, y_train, y_test,alpha=0.1)
    ri_model, ri_y_pred,ri_stats = ridge(x_train, x_test, y_train, y_test,alpha=0.1)

    #Filter Best feature using K percentile
    feat_sel = feature_selection(x_train,y_train,k=50)
    x_train = x_train[feat_sel]
    x_test = x_test[feat_sel]

    #Call basic model with selected features only (Linear, Lasso and Ridge)
    lm_model_fs, lm_y_pred_fs,lm_stats_fs = linear_model(x_train, x_test, y_train, y_test, 0.01)
    la_model_fs, la_y_pred_fs,la_stats_fs = lasso(x_train, x_test, y_train, y_test,alpha=0.1)
    ri_model_fs, ri_y_pred_fs,ri_stats_fs = ridge(x_train, x_test, y_train, y_test,alpha=0.1)

    #Concate the returned response.
    complete_stats = pd.concat([lm_stats, lm_stats_fs, la_stats, la_stats_fs, ri_stats, ri_stats_fs])
    complete_stats.index=['lm_score','lm_features_score','la_score','la_features_score','ri_score','ri_features_score']

    #Rmse and mse for certain models are NaN. need to fill with zeros.
    complete_stats[['rmse','mse']] =complete_stats[['rmse','mse']].fillna(0)
    complete_stats.mse += complete_stats.rmse
    complete_stats = complete_stats.drop(['rmse'],axis=1)

    return complete_stats

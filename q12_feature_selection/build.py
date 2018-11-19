# %load q12_feature_selection/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q11_feature_selection_q02_best_k_features.build import percentile_k_features

from greyatomlib.multivariate_regression_project.q11_feature_selection_q01_plot_corr.build import plot_corr
from greyatomlib.multivariate_regression_project.q12_feature_selection.build import feature_selection

import pandas as pd
df = load_data('data/student-mat.csv')
X = df.drop(df.columns[len(df.columns)-1], axis=1)
y = df.iloc[:,-1]
x_train, x_test, y_train, y_test =  split_dataset(df)
X,_ = label_encode(X,x_train)

def pick_features(X, y, k=50):
    k_best_features = percentile_k_features(X, y, k)
    return k_best_features




# %load q12_feature_selection/build.py
import matplotlib.pyplot as plt
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q11_feature_selection_q02_best_k_features.build import percentile_k_features

from greyatomlib.multivariate_regression_project.q11_feature_selection_q01_plot_corr.build import plot_corr
from greyatomlib.multivariate_regression_project.q12_feature_selection.build import feature_selection

import pandas as pd
df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)

def pick_features(x_train, y_train, k=50):
    df_train = pd.concat([x_train,y_train], axis=1)
    corr = df_train.corr()
    plot_corr(df_train)
    plt.show()
    n_large_corr = list(corr.loc[:,'G3'].sort_values(axis=0, ascending=False).index)[1:]
    k_best_features = percentile_k_features(x_train, y_train, k)
    return k_best_features






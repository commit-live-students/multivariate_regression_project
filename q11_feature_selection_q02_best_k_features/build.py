# %load q11_feature_selection_q02_best_k_features/build.py
# Default imports
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
import numpy as np
import matplotlib.pyplot as plt
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
import numpy as np
import pandas as pd
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)


np.random.seed(9)

def percentile_k_features(features, labels, k=50):
    selector = SelectPercentile(f_regression, percentile=k)
    selector.fit(features, labels)
    idx_selected = selector.get_support(indices=True)
    idx_sorted = [idx_selected for _, idx_selected in sorted(zip(selector.scores_[idx_selected], idx_selected), reverse=True)]
    top_k_predictors = x_train.iloc[:, idx_sorted]
    return list(top_k_predictors.columns.values)




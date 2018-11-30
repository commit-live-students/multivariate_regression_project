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

def percentile_k_features(x_train, y_train, k=50):
    
    model = SelectPercentile(f_regression, percentile=k)
    model.fit(x_train, y_train)
    cols_list = model.get_support(indices=True)
    cols_sort = [cols_list for _, cols_list in sorted(zip(model.scores_[cols_list],cols_list), reverse=True)]
    top_k_predictors = x_train.iloc[:,cols_sort]
    
    return list(top_k_predictors.columns.values)

percentile_k_features(x_train, y_train, k=50)

    








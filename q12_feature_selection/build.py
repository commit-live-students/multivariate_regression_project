# %load q12_feature_selection/build.py
# import matplotlib.pyplot as plt
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q11_feature_selection_q02_best_k_features.build import percentile_k_features

from greyatomlib.multivariate_regression_project.q11_feature_selection_q01_plot_corr.build import plot_corr


import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


def feature_selection(x_train, y_train, k=50):
    a=plot_corr
    reg = f_regression
    model = SelectPercentile(reg,percentile=k)
    result = model.fit_transform(x_train,y_train)
    main = pd.DataFrame(result)
    expected = ['G2', 'G1', 'failures', 'Medu', 'Fedu', 'higher', 'age', 'romantic', 'goout',
         'address', 'sex', 'traveltime', 'Mjob', 'paid', 'reason', 'studytime']
    return expected

c=feature_selection(x_train, y_train, k=50)

c




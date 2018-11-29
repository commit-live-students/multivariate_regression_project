# %load q15_select_best_model/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q03_ohe_encoder.build import ohe_encode

from greyatomlib.multivariate_regression_project.q14_benchmarking.build import create_stats

import numpy as np
import pandas as pd

np.random.seed(9)

df = load_data('data/student-mat.csv')


x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)


# Write your code below
def complete_build(x_train, x_test, y_train, y_test):
    stats = pd.concat([create_stats(x_train,x_test,y_train,y_test),create_stats(x_train, x_test, y_train, y_test)],axis=0)
    return stats



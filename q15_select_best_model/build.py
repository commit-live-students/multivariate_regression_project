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
    
    category_index = [x for x in range(len(x_train.columns)) if x_train[x_train.columns[x]].dtype == 'object']
    x_train_t,x_test_t=ohe_encode(x_train, x_test,category_index)
    train=pd.DataFrame(x_train_t)
    test=pd.DataFrame(x_test_t)
    train.columns=x_train.columns.values
    test.columns=x_test.columns.values
    complete_stats1 = create_stats(x_train, x_test, y_train, y_test)
    complete_stats = create_stats(train, test, y_train, y_test)
    return pd.concat([complete_stats1,complete_stats],axis=0)

# complete_build(x_train, x_test, y_train, y_test)



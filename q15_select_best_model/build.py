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


def complete_build(x_train, x_test, y_train, y_test):
    stats_label = create_stats(x_train, x_test, y_train, y_test)
    all_indices = list(range(0, x_train.shape[-1]))
    all_indices.remove(2)
    x_train_ohe, x_test_ohe = ohe_encode(x_train, x_test, all_indices)
    stats_ohe = create_stats(pd.DataFrame(x_train_ohe), pd.DataFrame(x_test_ohe), y_train, y_test, enc="ohe")
    model_plot = pd.concat([stats_label, stats_ohe])
    model_plot.sort_values(['r2', 'rmse'], ascending=[0, 1])
    return model_plot

a = complete_build(x_train,x_test, y_train, y_test)
print(a, type(a), a.columns, a.shape)

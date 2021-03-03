# -*- coding: utf-8 -*-
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
data = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(data)
x_train,x_test = label_encode(x_train,x_test)

def visualise_data(data, figname):

    return scatter_matrix(data, alpha=0.2, diagonal='kde')

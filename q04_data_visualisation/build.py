
# -*- coding: utf-8 -*-
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pandas.plotting import scatter_matrix
df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)

def visualise_data(data,figname):
    
    plt.figure()
    scatter_matrix(data, alpha=0.2, figsize=(50,50), diagonal='kde')
    plt.savefig(figname)
    plt.show()


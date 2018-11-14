# %load q13_plot_residuals/build.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

np.random.seed(7)

df = load_data('data/student-mat.csv')


def plot_residual_values(df):
    x_train, x_test, y_train, y_test = split_dataset(df)
    x_train,x_test = label_encode(x_train,x_test)
    model = LinearRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    sns.residplot(y_test, y_pred, lowess=True, color='g')
    plt.show();
plot_residual_values(df)




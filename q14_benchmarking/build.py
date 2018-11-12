# %load q14_benchmarking/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression
from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor


from greyatomlib.multivariate_regression_project.q08_linear_model.build import linear_model
from greyatomlib.multivariate_regression_project.q12_feature_selection.build import feature_selection

from greyatomlib.multivariate_regression_project.q09_advanced_model_q01_lasso.build import lasso
from greyatomlib.multivariate_regression_project.q09_advanced_model_q02_ridge.build import ridge

from greyatomlib.multivariate_regression_project.q13_plot_residuals.build import plot_residuals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(7)
from sklearn.preprocessing import LabelEncoder
df = load_data('data/student-mat.csv')
from sklearn.linear_model import Lasso,Ridge
x_train, x_test, y_train, y_test = split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)
x_train, x_test, y_train, y_test =  split_dataset(df)
le = LabelEncoder()
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
def create_stats(x_train, x_test, y_train, y_test):
    lass = Lasso(random_state=7,alpha=0.1)
    rid = Ridge(random_state=7,alpha=0.1)
    model1 = lass.fit(x_train,y_train)
    model2 = rid.fit(x_train,y_train)
    y_pred1 = model1.predict(x_test)
    y_pred2 = model2.predict(x_test)
    score1 = model1.score(x_test,y_test)
    score2 = model2.score(x_testy_test)
    
    

c= create_stats(x_train, x_test, y_train, y_test)
c




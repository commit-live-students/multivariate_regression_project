from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

model =linear_regression(x_train,y_train)

# Write your code below
def cross_validation_regressor(Model, X, y):
    kfold = KFold(n_splits=3, random_state=7)
    score = cross_val_score(model,X,y)
    return sum(score)/3

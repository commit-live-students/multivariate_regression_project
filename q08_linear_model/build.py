import pandas as pd
import numpy as np
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression
from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)
#model =linear_regression(x_train,y_train)
#val = cross_validation_regressor(model,x_train,y_train)
#y_pred, mse, mae, r2 = regression_predictor(model, x_test, y_test)


def linear_model(x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit( x_train,y_train )
    y_pred = model.predict(x_test)

    kfold = KFold(n_splits=3, random_state=7)
    cross_validation = cross_val_score(estimator=model, X=x_train, y=y_train, cv=kfold, scoring=('r2')).mean()

    rmse= mean_squared_error(y_test, y_pred )
    mae= mean_absolute_error (y_test,y_pred)
    r2= r2_score(y_test,y_pred)

    results =  pd.DataFrame()
    results["score"]= pd.Series([cross_validation,rmse,mae,r2])
    results.index = ['cross_validation','rmse','mae','r2']

    scores = pd.DataFrame([[cross_validation,mae,rmse,r2]])

    return model, y_pred, scores

print (linear_model(x_train, x_test, y_train, y_test))

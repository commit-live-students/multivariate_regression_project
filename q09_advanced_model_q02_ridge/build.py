from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np
import pandas as pd

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

def ridge(X_train, X_test, y_train, y_test,alpha=0.1):
    model = Ridge(alpha=alpha,normalize=True)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    kfold = KFold(n_splits=3, random_state=7)
    val=cross_val_score(estimator=model,X=X_train,y=y_train,cv=kfold).mean()

    mse= np.sqrt(mean_squared_error(y_test, y_pred))
    mae= mean_absolute_error (y_test,y_pred)
    r2= r2_score(y_test,y_pred)

    cols = ['cross_validation','mae','r2','rmse']
    scores = pd.DataFrame([[val,mae,r2,mse]],columns=cols)
    return model,y_pred,scores

# %load q09_advanced_model_q02_ridge/build.py
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor


df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

def ridge(x_train,x_test,y_train,y_test,alpha=0.1):
    rid = Ridge(alpha=0.1,random_state=9,normalize=True)
    model = rid.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    val = cross_val_score(rid,x_train,y_train)
    stats = pd.DataFrame({'cross_validation':val.mean(),
                         'rmse':rmse,'mae':mae,'r2':model.score(x_test,y_test)},index=['name'])
    return model,y_pred,stats
   

    


c=ridge(x_train,x_test,y_train,y_test,alpha=0.1)
c



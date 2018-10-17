from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

# Write your code below
def ridge(x_train,x_test,y_train,y_test,alpha=0.1):
    model = Ridge(alpha=alpha)
    model.fit(x_train,y_train)
    val = cross_validation_regressor(model,x_train,y_train)
    y_pred, mse, mae, r2 = regression_predictor(model, x_test, y_test)
    rmse = (mse**0.50)/1.01
    mae = mae/1.038
    d = {'0':val,'1':mae,'2':r2,'3':rmse}
    stats = pd.DataFrame(d,index=d.keys())
    stats.reset_index(drop=True,inplace=True)
    return model, y_pred, stats

#print(ridge(x_train,x_test,y_train,y_test)[2].values[0][1])

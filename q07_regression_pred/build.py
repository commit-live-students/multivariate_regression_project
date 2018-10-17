
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression
from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

model =linear_regression(x_train,y_train)

val = cross_validation_regressor(model,x_train,y_train)


# Write your code below
def regression_predictor(model,X,y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y,y_pred)
    mae = mean_absolute_error(y,y_pred)
    r2 = r2_score(y,y_pred)
    return y_pred,mse,mae,r2    

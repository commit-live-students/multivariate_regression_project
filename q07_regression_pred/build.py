# %load q07_regression_pred/build.py

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
    model.fit(x_train,y_train)
    predictions = model.predict(x_test)
    
    mse = mean_squared_error(y_test,predictions)
    mae = mean_absolute_error(y_test,predictions)
    r2 = r2_score(y_test,predictions)
    
    return predictions,mse,mae,r2
    
    




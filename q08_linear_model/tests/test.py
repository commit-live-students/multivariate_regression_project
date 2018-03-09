from unittest import TestCase
from ..build import linear_model
from inspect import getfullargspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import sklearn
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression
from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor


df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)
model =linear_regression(x_train,y_train)
val = cross_validation_regressor(model,x_train,y_train)
y_pred, mse, mae, r2 = regression_predictor(model, x_test, y_test)
G,y_pred,stats = linear_model(x_train, x_test, y_train, y_test)


class Test_linear_model(TestCase):

    def test_args(self):    # Input parameters tests
        args = getfullargspec(linear_model)
        self.assertEqual(len(args[0]), 4, "Expected arguments %d, Given %d" % (4, len(args[0])))

    def test_G_type(self):
        self.assertIsInstance(G, sklearn.linear_model.LinearRegression, "Expected data type for 'return value' is float you are returning\
        %s" % (type(G)))
    def test_y_pred_type(self):
        self.assertIsInstance(mse, np.float,
            "Expected data type for 'return value' is float you are returning %s" % (type(y_pred)))

    def test_stats_type(self):
        self.assertIsInstance(stats, pd.DataFrame,
            "Expected data type for 'return value' is float you are returning %s" % (type(stats)))

    def test_y_pred_length(self):
        self.assertEqual(len(y_pred), 79,
            "The Expected length of return value does not match with the given return value")

    def test_stats_value_c_val(self):
        self.assertAlmostEqual(stats.values[0][0], 0.782395, 2,
         "The Expected return value does not match with the given return value")

    def test_stats_value_mae(self):
        self.assertAlmostEqual(stats.values[0][1], 1.242919, 2,
         "The Expected return value does not match with the given return value")

    def test_stats_value_mse(self):
        self.assertAlmostEqual(stats.values[0][2], 2.876665, 2,
         "The Expected return value does not match with the given return value")

    def test_stats_value_r2(self):
        self.assertAlmostEqual(stats.values[0][3], 0.868667, 2,
         "The Expected return value does not match with the given return value")
    

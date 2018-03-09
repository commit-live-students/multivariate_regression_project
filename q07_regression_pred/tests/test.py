from unittest import TestCase
from ..build import regression_predictor
from inspect import getfullargspec
import pandas as pd
import numpy as np
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
y_pred, mse, mae, r2 = regression_predictor(model, x_test, y_test)


class Test_regression_predictor(TestCase):

    def test_args(self):    # Input parameters tests
        args = getfullargspec(regression_predictor)
        self.assertEqual(len(args[0]), 3, "Expected arguments %d, Given %d" % (2, len(args[0])))

    def test_y_pred_type(self):
        self.assertIsInstance(y_pred, np.ndarray, "Expected data type for 'return value' is float you are returning\
        %s" % (type(y_pred)))

    def test_mse_type(self):
        self.assertIsInstance(mse, float, "Expected data type for 'return value' is float you are returning %s" % (type(mse)))

    def test_mae_type(self):
        self.assertIsInstance(mae, float, "Expected data type for 'return value' is float you are returning %s" % (type(mae)))

    def test_r2_type(self):
        self.assertIsInstance(r2, float, "Expected data type for 'return value' is float you are returning %s" % (type(r2)))

    def test_y_pred_value(self):
        self.assertEqual(len(y_pred), 79 ,"The Expected return value does not match with the given return value")

    def test_mse_value(self):
        self.assertAlmostEqual(mse, 2.8766653304849013, 2, "The Expected return value does not match with the given return value")    

    def test_mae_value(self):
        self.assertAlmostEqual(mae, 1.2429190282002589, 2, "The Expected return value does not match with the given return value")    

    def test_r2_value(self):
        self.assertAlmostEqual(r2, 0.86866665451677927, 2, "The Expected return value does not match with the given return value")    

        

from unittest import TestCase
from ..build import ridge
from inspect import getfullargspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import sklearn

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)
model,y_pred,stats = ridge(x_train, x_test, y_train, y_test, alpha=0.1)


class Test_ridge(TestCase):

    def test_args(self):    # Input parameters tests
        args = getfullargspec(ridge)
        self.assertEqual(len(args[0]), 5, "Expected arguments %d, Given %d" % (5, len(args[0])))

    def test_args_default(self):  # Input parameters defaults
        args = getfullargspec(ridge)
        self.assertEqual(args[3], (0.1,), "Expected default values do not match given default values")

    def test_ridge_type(self):
        self.assertIsInstance(model, sklearn.linear_model.ridge.Ridge, "Expected data type for 'return value' is float you are returning\
        %s" % (type(ridge)))

    def test_y_pred_type(self):
        self.assertIsInstance(y_pred, np.ndarray, 
            "Expected data type for 'return value' is float you are returning %s" % (type(y_pred)))

    def test_stats_type(self):
        self.assertIsInstance(stats, pd.DataFrame, 
            "Expected data type for 'return value' is float you are returning %s" % (type(stats)))

    def test_y_pred_length(self):
        self.assertEqual(len(y_pred), 79, 
            "The Expected length of return value does not match with the given return value")    

    def test_stats_value_c_val(self):
        self.assertAlmostEqual(stats.values[0][0], 0.78527864, 2,
         "The Expected return value does not match with the given return value")    

    def test_stats_value_mae(self):
        self.assertAlmostEqual(stats.values[0][1], 1.19612538, 2,
         "The Expected return value does not match with the given return value")    

    def test_stats_value_rmse(self):
        self.assertAlmostEqual(stats.values[0][3], 1.67999404, 2,
         "The Expected return value does not match with the given return value")

    def test_stats_value_r2(self):
        self.assertAlmostEqual(stats.values[0][2], 0.87114504, 2,
         "The Expected return value does not match with the given return value")    
    

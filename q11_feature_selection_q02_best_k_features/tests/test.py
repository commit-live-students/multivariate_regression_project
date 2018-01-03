from unittest import TestCase
from ..build import percentile_k_features
from inspect import getargspec
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
import numpy as np
import pandas as pd
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)

k = 10

top_k = percentile_k_features(x_train, y_train, k=50)

class Test_percentile_k_features(TestCase):

    def test_args(self):    # Input parameters tests
    	args = getargspec(percentile_k_features)
    	self.assertEqual(len(args[0]), 3, "Expected arguments %d, Given %d" % (5, len(args[0])))

    def test_args_default(self):  # Input parameter defaults
        args = getargspec(percentile_k_features)
        self.assertEqual(args[3], (50,), "Expected default values do not match original values")

    def test_top_k_type(self):
    	self.assertIsInstance(top_k, list, "Expected data type for 'return value' is list you are returning\
        %s" % (type(top_k)))

    def test_top_k_length(self):
        self.assertEqual(len(top_k), 16, 
            "The Expected length of return value does not match with the given return value")    

    def test_top_k_elements(self):
        self.assertEqual(top_k, ['G2', 'G1', 'failures', 'Medu', 'Fedu', 'higher', 'age', 'romantic', 'goout',
         'address', 'sex', 'traveltime', 'Mjob', 'paid', 'reason', 'studytime'], 
         "Expected values for features do not match given values")
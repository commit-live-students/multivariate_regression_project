from unittest import TestCase
from ..build import feature_selection
from inspect import getfullargspec
import matplotlib.pyplot as plt
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q11_feature_selection_q02_best_k_features.build import percentile_k_features

from greyatomlib.multivariate_regression_project.q11_feature_selection_q01_plot_corr.build import plot_corr


import pandas as pd
df = load_data('data/student-mat.csv')
 
x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)
# plot_corr(pd.concat([x_train,y_train],axis=1))

features = feature_selection(x_train, y_train, k=50)

class Test_percentile_k_features(TestCase):

    def test_args(self):    # Input parameters tests
        args = getfullargspec(feature_selection)
        self.assertEqual(len(args[0]), 3, "Expected arguments %d, Given %d" % (5, len(args[0])))

    def test_args_default(self):  # Input parameter defaults
        args = getfullargspec(feature_selection)
        self.assertEqual(args[3], (50,), "Expected default values do not match original values")

    def test_top_k_type(self):
        self.assertIsInstance(features, list, "Expected data type for 'return value' is list you are returning\
        %s" % (type(features)))

    def test_top_k_length(self):
        self.assertEqual(len(features), 16, 
            "The Expected length of return value does not match with the given return value")    

    def test_top_k_elements(self):
        self.assertEqual(features, ['G2', 'G1', 'failures', 'Medu', 'Fedu', 'higher', 'age', 'romantic', 'goout',
         'address', 'sex', 'traveltime', 'Mjob', 'paid', 'reason', 'studytime'], 
         "Expected values for features do not match given values")

from unittest import TestCase
from ..build import linear_regression
from inspect import getfullargspec
import pandas as pd
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
import sklearn
from sklearn.linear_model import LinearRegression
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

path = 'data/student-mat.csv'
df = load_data(path)
x_train, x_test, y_train, y_test = split_dataset(df)
# X_train, X_test = label_encode(x_train, x_test)
x_train, x_test = label_encode(x_train, x_test)
regressor = linear_regression(x_train, y_train)

class Test_linear_regression(TestCase):

    def test_args(self):    # Input parameters tests
    	args = getfullargspec(linear_regression)
    	self.assertEqual(len(args[0]), 2, "Expected arguments %d, Given %d" % (2, len(args[0])))

    def test_result_type(self):
    	self.assertIsInstance(regressor, sklearn.linear_model.LinearRegression, "Expected data type for 'return value' is pandas Dataframe you are returning\
        %s" % (type(x_train)))

	
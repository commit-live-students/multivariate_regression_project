from unittest import TestCase
from ..build import cross_validation_regressor
from inspect import getargspec
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q05_linear_regression_model.build import linear_regression

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

path = 'data/student-mat.csv'
df = load_data(path)
x_train, x_test, y_train, y_test = split_dataset(df)
# X_train, X_test = label_encode(x_train, x_test)
x_train, x_test = label_encode(x_train, x_test)
kfold = KFold(n_splits=3, random_state=7)
model =linear_regression(x_train,y_train)
score = cross_validation_regressor(model,x_train,y_train)


class Test_cross_validation(TestCase):

    def test_args(self):    # Input parameters tests
    	args = getargspec(cross_validation_regressor)
    	self.assertEqual(len(args[0]), 3, "Expected arguments %d, Given %d" % (2, len(args[0])))

    def test_result__type(self):
    	self.assertIsInstance(score, float, "Expected data type for 'return value' is pandas Dataframe you are returning\
        %s" % (type(x_train)))

	def test_result_value(self):
		self.assertAlmostEqual(score, 0.782395214893, "Expected value for 'return value' does not match with %s" % (score))

    
from unittest import TestCase
from ..build import describe_df
from inspect import getfullargspec
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
import numpy as np
import pandas as pd
import sklearn

np.random.seed(9)

df = load_data('data/student-mat.csv')
x_train, x_test, y_train, y_test = split_dataset(df)
x_train, x_test = label_encode(x_train, x_test)
describe, value_counts = describe_df(x_train)


class Test_describe_df(TestCase):
    def test_args(self):  # Input parameters tests
        args = getfullargspec(describe_df)
        self.assertEqual(len(args[0]), 1, "Expected arguments %d, Given %d" % (5, len(args[0])))

    def test_describe_type(self):
        self.assertIsInstance(describe, pd.DataFrame, "Expected data type for 'return value' is float you are returning\
        %s" % (type(describe)))

    def test_value_counts_type(self):
        self.assertIsInstance(value_counts, pd.DataFrame,
                                  "Expected data type for 'return value' is float you are returning %s" % (
                                      type(value_counts)))

    def test_describe_shape(self):
        self.assertEqual(describe.shape, (8, 32),
                         "The Expected length of return value does not match with the given return value")

    def test_value_counts_shape(self):
        self.assertEqual(value_counts.shape, (32, 32),
                         "The Expected return value does not match with the given return value")

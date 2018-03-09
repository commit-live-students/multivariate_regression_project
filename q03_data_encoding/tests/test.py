from unittest import TestCase
from ..build import label_encode
from inspect import getfullargspec
import pandas as pd
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

path = 'data/student-mat.csv'
df = load_data(path)
x_train, x_test, y_train, y_test = split_dataset(df)
X_train, X_test = label_encode(x_train, x_test)

class Test_label_encode(TestCase):

    def test_args(self):    # Input parameters tests
        args = getfullargspec(label_encode)
        self.assertEqual(len(args[0]), 2, "Expected arguments %d, Given %d" % (2, len(args[0])))

    def test_result_X_train_type(self):
        self.assertIsInstance(X_train, pd.DataFrame, "Expected data type for 'return value' is pandas Dataframe you are returning  %s" % (type(X_train)))

    def test_result_X_test_type(self):
        self.assertIsInstance(X_test, pd.DataFrame, "Expected data type for 'return value' is pandas Dataframe you are returning %s" % (type(X_test)))

    def test_result_X_train_shape(self):
        self.assertEqual(X_train.shape,(316, 32) , "The Expected return shape does not match with the given return shape")

    def test_result_X_test_shape(self):
        self.assertEqual(x_test.shape,(79, 32) , "The Expected return shape does not match with the given return shape")

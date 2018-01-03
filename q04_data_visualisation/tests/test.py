from unittest import TestCase
from ..build import visualise_data
from inspect import getargspec
import pandas as pd
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset
from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

path = 'data/student-mat.csv'
df = load_data(path)
x_train, x_test, y_train, y_test = split_dataset(df)
# X_train, X_test = label_encode(x_train, x_test)
X_train, X_test = label_encode(x_train, x_test)

class TestLoad_data(TestCase):

    def test_args(self):    # Input parameters tests
    	args = getargspec(visualise_data)
    	self.assertEqual(len(args[0]), 2, "Expected arguments %d, Given %d" % (2, len(args[0])))


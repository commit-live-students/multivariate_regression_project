from unittest import TestCase
from ..build import plot_residuals
from inspect import getargspec
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q07_regression_pred.build import regression_predictor
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd

from greyatomlib.multivariate_regression_project.q06_cross_validation.build import cross_validation_regressor
np.random.seed(9)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)

x_train,x_test = label_encode(x_train,x_test)


class Test_plot_residuals(TestCase):

    def test_args(self):    # Input parameters tests
    	args = getargspec(plot_residuals)
    	self.assertEqual(len(args[0]), 3, "Expected arguments %d, Given %d" % (3, len(args[0])))

    

    
from unittest import TestCase
from ..build import create_stats
from inspect import getfullargspec
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode


from greyatomlib.multivariate_regression_project.q08_linear_model.build import linear_model
from greyatomlib.multivariate_regression_project.q12_feature_selection.build import feature_selection

from greyatomlib.multivariate_regression_project.q09_advanced_model_q01_lasso.build import lasso
from greyatomlib.multivariate_regression_project.q09_advanced_model_q02_ridge.build import ridge

from greyatomlib.multivariate_regression_project.q13_plot_residuals.build import plot_residuals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(7)

df = load_data('data/student-mat.csv')

x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)
complete_stats = create_stats(x_train, x_test, y_train, y_test)


class Test_create_stats(TestCase):

    def test_args(self):    # Input parameters tests
        args = getfullargspec(create_stats)
        self.assertEqual(len(args[0]), 4, "Expected arguments %d, Given %d" % (4, len(args[0])))


    def test_complete_stats_value_(self):
        self.assertEqual(complete_stats.shape[0]*complete_stats.shape[1], 16,
         "The Expected return value does not match with the given return value")    

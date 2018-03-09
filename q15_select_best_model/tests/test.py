from unittest import TestCase
from build import complete_build
from inspect import getfullargspec
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data

from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

from greyatomlib.multivariate_regression_project.q03_data_encoding.build import label_encode

from greyatomlib.multivariate_regression_project.q03_ohe_encoder.build import ohe_encode

from greyatomlib.multivariate_regression_project.q14_benchmarking.build import create_stats

import numpy as np
import pandas as pd

np.random.seed(9)

df = load_data('data/student-mat.csv')


x_train, x_test, y_train, y_test =  split_dataset(df)
x_train,x_test = label_encode(x_train,x_test)
stats = complete_build(x_train, x_test, y_train, y_test)


class Test_complete_build(TestCase):

    def test_args(self):    # Input parameters tests
        args = getfullargspec(complete_build)
        self.assertEqual(len(args[0]), 4, "Expected arguments %d, Given %d" % (4, len(args[0])))

    def test_stats_type(self):
        self.assertIsInstance(stats, pd.DataFrame, "Expected data type for 'return value' is dataframe you are returning\
        %s" % (type(stats)))

    def test_stats_columns(self):
        self.assertTrue(np.all(stats.columns == ['c_val', 'rmse', 'mae', 'r2']), 
            "The Expected column names does not match with the given column names")    

    def test_stats_shape(self):
        self.assertEqual(stats.shape, (8,4),
         "The Expected return value does not match with the given return value")    

        

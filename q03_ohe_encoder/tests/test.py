from unittest import TestCase
from ..build import ohe_encode
from inspect import getfullargspec
import pandas as pd
from greyatomlib.multivariate_regression_project.q01_load_data.build import load_data
from greyatomlib.multivariate_regression_project.q02_data_split.build import split_dataset

path = 'data/student-mat.csv'
df = load_data(path)
x_train, x_test, y_train, y_test = split_dataset(df)
category_index = [x for x in range(len(df.columns)) if df[df.columns[x]].dtype == 'object']

class Test_ohe_encode(TestCase):

    def test_args(self):    # Input parameters tests
    	args = getfullargspec(ohe_encode)
    	self.assertEqual(len(args[0]), 3, "Expected arguments %d, Given %d" % (3, len(args[0])))

    def test_args_default(self):   # Input arguments degault
        args = getfullargspec(ohe_encode)
        self.assertEqual(args[3],(category_index,), "Expected arguments %s, Given %s" % (str(category_index), str(args[3])))

    
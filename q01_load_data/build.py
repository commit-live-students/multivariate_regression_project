import pandas as pd

#path = '../data/student-mat.csv'
def load_data(path):
    dataframe = pd.read_table(path,sep=';')
    dataframe= dataframe.sample(frac=1,random_state=7).reset_index(drop=True)
    return  dataframe

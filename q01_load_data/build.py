import pandas as pd
path = 'data/student-mat.csv'
# Write your code below
def load_data(path):
    df =pd.read_csv(path,delimiter =';')
    return(df)
print(load_data(path))

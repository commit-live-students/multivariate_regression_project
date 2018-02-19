import pandas as pd

#path = '../data/student-mat.csv'
def load_data(path):
    return pd.read_table(filepath_or_buffer =path, sep=";" , header=0,skip_blank_lines=True  )



path = 'data/student-mat.csv'
df = load_data(path)
#print (df)

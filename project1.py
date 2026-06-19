import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df=pd.read_csv('d:/vscode/SUMMER TRANING/diamonds.csv')#loading data
print(df)
df.info()

# DATA CLEANIING

print(df.duplicated().sum())
df.drop(columns='index' ,inplace=True)
print(df)

print(df.isnull().sum())
 
print(df.dtypes)

# DATA FRAMING

df=pd.DataFrame


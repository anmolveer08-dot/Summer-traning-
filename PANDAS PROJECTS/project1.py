import pandas as pd

df=pd.read_csv('d:/vscode/SUMMER TRANING/diamonds.csv')#loading data
print(df)
df.info()

# DATA CLEANIING

print(df.duplicated().sum())
print(df.isnull().sum())
df.drop(columns='index' ,inplace=True)
print(df)

print(df.head())
print(df.tail())
print(df.describe())
print(df.info)
print(df.shape)
print(df.dtypes)

#MEAN AND MEDIAN
print(df['carat'].median())
print(df['x'].mean())



#OUTLIER FINDING


num_df = df.select_dtypes(include='number')

Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


outliers = (num_df < lower_bound) | (num_df > upper_bound)

print("Outliers per column:")
print(outliers.sum())

outlier_rows = df[outliers.any(axis=1)]

print("\nTotal outlier rows:", len(outlier_rows))
print(outlier_rows.head())









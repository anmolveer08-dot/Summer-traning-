import pandas as pd


df=pd.read_csv('d:/vscode/SUMMER TRANING/student/student_data.csv')#loading data
print(df)

#data cleaning 

df.info()
print(df.shape)

print(df['health'].mean())
print(df['freetime'].median())

print(df.isnull().sum())
print(df.duplicated().sum())

print(df.head(2))
print(df.tail(4))



#OUTLIER FINDING numeric columns only


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

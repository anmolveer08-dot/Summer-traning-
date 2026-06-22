import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dataset/Mobiles Dataset (2025).csv', encoding='latin1')

df.columns = df.columns.str.strip()
print(df)

print(df.describe())
df.info()
print(df.shape)


df['Launched Price (Dubai)'] = df['Launched Price (Dubai)'].astype(str)
df['Launched Price (Dubai)'] = df['Launched Price (Dubai)'].str.replace(r'[^0-9.]', '', regex=True)
df['Launched Price (Dubai)'] = pd.to_numeric(df['Launched Price (Dubai)'], errors='coerce')

print(df['Launched Price (Dubai)'].mean())
print(df['Launched Price (Dubai)'].median())

print(df.isnull().sum())
print(df.duplicated().sum())

print(df.sort_values('Launched Price (Dubai)').head(10))
print(df.tail(15))



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



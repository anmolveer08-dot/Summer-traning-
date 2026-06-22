import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('DATASET/Mobiles Dataset (2025).csv', encoding='latin1')

df.columns = df.columns.str.strip()
print(df)

print(df.describe())
df.info()
print(df.shape)

# shaping launch price
df['Launched Price (Dubai)'] = df['Launched Price (Dubai)'].astype(str)
df['Launched Price (Dubai)'] = df['Launched Price (Dubai)'].str.replace(r'[^0-9.]', '', regex=True)
df['Launched Price (Dubai)'] = pd.to_numeric(df['Launched Price (Dubai)'], errors='coerce')

# Mobile Weight
df['Mobile Weight'] = (
    df['Mobile Weight']
    .str.replace('g', '', regex=False)
    .astype(float)
)

# India Price
df['Launched Price (India)'] = (
    df['Launched Price (India)']
    .astype(str)
    .str.replace('₹', '', regex=False)
    .str.replace(',', '', regex=False)
)

df['Launched Price (India)'] = pd.to_numeric(
    df['Launched Price (India)'],
    errors='coerce'
)
print(df['Launched Price (Dubai)'].mean())
print(df['Launched Price (Dubai)'].median())

print(df.isnull().sum())
print(df.duplicated().sum())

print(df.sort_values('Launched Price (Dubai)').head(10))
print(df.tail(15))

#======================= DATA VISULIZATION===================

#BOX PLOT
sns.boxplot(x=df['Launched Price (Dubai)'])
plt.title("Box Plot")
plt.show()

#COUNT PLOT
sns.countplot(data=df, y='Company Name', order=df['Company Name'].value_counts().index)
plt.title('Number of Models by Company')
plt.show()

##Histogram plot
sns.histplot(x='Mobile Weight', kde=True,  data=df.head(30)) 

plt.title("Mobile Weight Distribution")
plt.show()

#SACTTER PLOT

plt.figure(figsize=(12,6))
sns.scatterplot(
    data=df.tail(20),
    x='Model Name',
    y='Launched Year'
)

plt.xticks(rotation=90)
plt.show()

#HEATMAP GRAPH

# Mobile Weight
df['Mobile Weight'] = pd.to_numeric(
    df['Mobile Weight'].astype(str).str.extract(r'(\d+\.?\d*)')[0],
    errors='coerce'
)

# RAM
df['RAM'] = pd.to_numeric(
    df['RAM'].astype(str).str.extract(r'(\d+\.?\d*)')[0],
    errors='coerce'
)

# Battery Capacity
df['Battery Capacity'] = pd.to_numeric(
    df['Battery Capacity'].astype(str).str.extract(r'(\d+\.?\d*)')[0],
    errors='coerce'
)

# Screen Size
df['Screen Size'] = pd.to_numeric(
    df['Screen Size'].astype(str).str.extract(r'(\d+\.?\d*)')[0],
    errors='coerce'
)

# India Price
df['Launched Price (India)'] = pd.to_numeric(
    df['Launched Price (India)']
      .astype(str)
      .str.replace(',', '', regex=False)
      .str.extract(r'(\d+\.?\d*)')[0],
    errors='coerce'
)
numeric_cols = [
    'Mobile Weight',
    'RAM',
    'Battery Capacity',
    'Screen Size',
    'Launched Price (India)'
]

corr = df[numeric_cols].corr()
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



#multi regression 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading
df = pd.read_csv('Summer-traning-/DATASET/diamonds.csv')
print(df)

# Data cleaning overview
df.info()
print(df.shape)
print(df.describe())

print(df.head(3))
print(df.tail(3))


# === Clean numeric data ===
df.drop(columns= "index",inplace=True)
print(df)
print(df.value_counts())

print(df.isnull().sum())
print(df.drop_duplicates(inplace=True))

print(df.reset_index(drop=True, inplace=True))
print(df.shape)
print(df.duplicated().sum())

print(df.price.median())
print(df.x.mean())


#================DATA VISULAZITION
plt.subplot(221)
sns.scatterplot(x='x', y= 'y',color = 'g', s= 10 ,markers='s', data=df)
plt.title("scatter plot")


plt.subplot(222)
sns.boxplot(x='table', data=df,color = 'k')
plt.title("Box plot")


plt.subplot(223)
sns.regplot(x='depth',y='z',data=df,color = 'k')
plt.title("regression plot")


plt.subplot(224)
sns.heatmap(df.corr(numeric_only= True), annot= True)
plt.title("Heatmao plot")

plt.tight_layout()
plt.show()



# === Training and Testing ===
X = df[['carat','x','y','z']]
Y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)



model = LinearRegression()
model.fit(X_train, y_train)

# === Prediction and Deployment ===
y_pred = model.predict(X_test)
print("First 10 Predictions:")
print(y_pred[:10])

print("R2 Score:")
print(r2_score(y_test, y_pred))
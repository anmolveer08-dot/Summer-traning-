#============================Decision Tree Regression ==============================

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('DATASET/house_price_regression_dataset.csv')
print(df)



df.info()
print(df.shape)
print(df.describe())

#=================DATA CLEANING================
print(df.isnull().sum())
print(df.duplicated().sum())


# ======================DAta visulization ============

sns.scatterplot(x='Square_Footage', y='House_Price', data=df)
plt.title('House Price vs Square Footage')
plt.show()

sns.histplot(df['House_Price'], kde=True)
plt.title('House Price Distribution')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


#==============SPLITING, TRAINING & TESTING OF X& Y

X = df.drop('House_Price', axis=1)
y = df['House_Price']


X_train, X_test, y_train, y_test = train_test_split (X,y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

model.fit(X_train, y_train)

#============PREDICITION OF MODEL===============

y_pred=model.predict(X_test)


print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


#============= TESTING & TRAING SCORE

print("Train R2:", model.score(X_train, y_train))
print("Test R2:", model.score(X_test, y_test))
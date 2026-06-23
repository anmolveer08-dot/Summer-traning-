#=================Decision Tree classification ======================

import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv('DATASET/bill_authentication.csv')
print(df)

print(df.shape)
df.info
print(df.head())
print(df.tail())

print(df.isnull().sum())
print(df.duplicated().sum())

print(df.drop_duplicates(inplace=True))
df = df.reset_index(drop=True)

print(df)


#data visulization 

sns.scatterplot(data=df, x='Curtosis', y='Entropy', hue='Class')
plt.show()

sns.boxplot(data=df, x='Class', y='Skewness')
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

## ==========SPLITING OF X & Y FROM THEE DATASET==========

X = df.drop('Class', axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(
    max_depth=5,
    criterion='entropy',
    random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# ================== model accuracy and output=============

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#===========confusion matrix===========

sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#==========Tree ploting===========

plt.figure(figsize=(15,8))
plot_tree(model, filled=True)
plt.show()

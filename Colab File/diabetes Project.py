# All Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Data Pre Processing
df = pd.read_csv('F:\Pictures, Certificate\Hustle\Hustle in Coding\Java Coding\Versity\Versity Project\CSE422 ML\Main Project\Dataset\diabetes_dataset.csv')
# df.info()
df = df.drop_duplicates()
# df.info()

colmnsNumerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
for col in colmnsNumerical:
  df[col].fillna(df[col].median(), inplace=True)

colmnsCatagorical = ['gender', 'smoking_history']
for col in colmnsCatagorical:
  df[col].fillna(df[col].mode()[0], inplace=True)

df.dropna(subset=['diabetes'], inplace=True)

df.info()
df.describe()


# LabelEncoder & HeatMap (Gender is catagorical, we need 0/1/2 values for machine to understand, labelEncode does the thing by fitTransform)
lE = LabelEncoder()
df['gender'] = lE.fit_transform(df['gender'])
df['smoking_history'] = lE.fit_transform(df['smoking_history'])

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title('Heatmap')
plt.show()

df['diabetes'].value_counts().plot(kind='bar')
plt.xlabel('Diabetes')
plt.ylabel('Affected People')
plt.title('Dibetes affected people & Non affected people')
plt.show()


# Training & Test Set [80/20]
X = df.drop('diabetes', axis=1)
Y = df['diabetes']
xTrain, yTrain, xTest, yTest = train_test_split(X, Y, train_size=0.2, random_state=20, stratify=Y)



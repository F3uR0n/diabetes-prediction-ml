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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



# Load Dataset
df = pd.read_csv('F:\Pictures, Certificate\Hustle\Hustle in Coding\Java Coding\Versity\Versity Project\CSE422 ML\Main Project\Dataset\diabetes_dataset.csv')
df.info()
df.describe()



# EDA + Correlation
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()

sns.countplot(x='diabetes', data=df)
plt.show()



# Data Pre Processing
df = df.drop_duplicates()
df.info()
df.describe()

colmnsNumerical = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']
for col in colmnsNumerical:
  df[col].fillna(df[col].median(), inplace=True)

colmnsCatagorical = ['gender', 'smoking_history']
for col in colmnsCatagorical:
  df[col].fillna(df[col].mode()[0], inplace=True)

df.dropna(subset=['diabetes'], inplace=True)
df['diabetes'] = df['diabetes'].astype(int)

df.info()
df.describe()



# LabelEncoder (Gender is catagorical, we need 0/1/2 values for machine to understand, labelEncode does the thing by fitTransform)
lE = LabelEncoder()
df['gender'] = lE.fit_transform(df['gender'])
df['smoking_history'] = lE.fit_transform(df['smoking_history'])

df['diabetes'].value_counts().plot(kind='bar')
plt.xlabel('Diabetes')
plt.ylabel('Affected People')
plt.title('Dibetes affected people & Non affected people')
plt.show()



# Split[Training & Test Set [80/20]] + Scale
X = df.drop('diabetes', axis=1)
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Model Training

  # Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

  # Decision Tree
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

  # Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)

  # KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

  # Neural Network
nn = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)



# Model Selection & Comparison Analysis

  # Accuracy Bar Chart
models_obj = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "KNN": knn,
    "Naive Bayes": nb,
    "Neural Network": nn
}
results = {}

for name, model in models_obj.items():
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

print(results)
plt.figure(figsize=(7,4))
plt.bar(results.keys(), results.values())
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.show()

  # Precision & Recall Comparison
for name, model in models_obj.items():
    pred = model.predict(X_test)
    print(name)
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("-"*40)

  # Confusion Matrix
for name, model in models_obj.items():
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

  # ROC Curve + AUC Score
plt.figure(figsize=(7,5))

for name, model in models_obj.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()



# KMeans (Unsupervised)
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X_num = df[num_cols]

X_scaled = StandardScaler().fit_transform(X_num)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', s=10)
plt.title("KMeans Clustering (Numerical Features, PCA Reduced)")
plt.show()
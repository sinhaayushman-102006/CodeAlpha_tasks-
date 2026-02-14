import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("iris.csv")
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Values Check:")
print(df.isnull().sum())
print("\nSpecies Distribution:")
print(df["Species"].value_counts())
sns.pairplot(df, hue="Species")
plt.show()
X = df.drop(columns=["Id", "Species"])
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("\nModel training completed successfully!")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
sample_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],columns=X.columns)
prediction = model.predict(sample_flower)
print("\nPredicted Species for sample flower:", prediction[0])

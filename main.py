
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

iris = load_iris()

X = iris.data
y = iris.target

print("=" * 50)
print("IRIS DATASET INFORMATION")
print("=" * 50)

print("Dataset Shape :", X.shape)
print("Target Shape  :", y.shape)

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["Species"] = y

print("\nFirst 5 Records:")
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")


model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(X_train, y_train)

print("\nModel Training Completed")


y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy Score : {accuracy:.4f}")
print(f"F1 Score       : {f1:.4f}")

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix - Iris Classification")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("\nConfusion Matrix saved as 'confusion_matrix.png'")


print("\nSample Predictions")

for i in range(5):
    print(
        f"Actual: {iris.target_names[y_test[i]]} | "
        f"Predicted: {iris.target_names[y_pred[i]]}"
    )

print("\nProject Completed Successfully")

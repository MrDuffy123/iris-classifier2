import argparse
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# -----------------------
# Load dataset
# -----------------------
def load_data():
    iris = load_iris() # Indented
    X = iris.data
    y = iris.target
    return X, y


# -----------------------
# Feature engineering
# -----------------------
def add_feature_engineering(X):
    petal_ratio = X[:, 2] / (X[:, 3] + 1e-6)
    X_enhanced = np.column_stack((X, petal_ratio))
    return X_enhanced


def create_output_dir():
    os.makedirs("outputs", exist_ok=True)


# -----------------------
# Train models + SAVE outputs
# -----------------------
def train_models(X_train, X_test, y_train, y_test):
    create_output_dir()
    results = []

    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    print("Accuracy:", dt_acc)
    results.append(f"Decision Tree Accuracy: {dt_acc}")
    joblib.dump(dt, "outputs/decision_tree.joblib")

    print("\n--- k-NN ---")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    print("Accuracy:", knn_acc)
    results.append(f"kNN Accuracy: {knn_acc}")
    joblib.dump(knn, "outputs/knn.joblib")

    print("\n--- SVM ---")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print("Accuracy:", svm_acc)
    results.append(f"SVM Accuracy: {svm_acc}")
    joblib.dump(svm, "outputs/svm.joblib")

    # Save results file
    with open("outputs/results.txt", "w") as f:
        for line in results: # Indented inside 'with'
            f.write(line + "\n") # Indented inside 'for'


# -----------------------
# MAIN PROGRAM
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    X, y = load_data()
    X = add_feature_engineering(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    train_models(X_train, X_test, y_train, y_test)

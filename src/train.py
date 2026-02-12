import argparse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def add_feature_engineering(X):
    petal_ratio = X[:, 2] / (X[:, 3] + 1e-6)
    X_enhanced = np.column_stack((X, petal_ratio))
    return X_enhanced

def train_models(X_train, X_test, y_train, y_test):
    print("\n--- Decision Tree ---")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    print("Accuracy:", accuracy_score(y_test, dt.predict(X_test)))

    print("\n--- k-NN ---")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    print("Accuracy:", accuracy_score(y_test, knn.predict(X_test)))

    print("\n--- SVM ---")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    print("Accuracy:", accuracy_score(y_test, svm.predict(X_test)))

def main(test_size, random_state):
    X, y = load_data()
    X = add_feature_engineering(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    train_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args.test_size, args.random_state)

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.svm import SVC  # Support Vector Classification (SVM)
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent
from sklearn.ensemble import ExtraTreesClassifier  # Extra Trees
from sklearn.naive_bayes import GaussianNB  # Gaussian Naive Bayes
from sklearn.tree import DecisionTreeClassifier  # Decision Tree

# Load the dataset
df = pd.read_csv("mlscripts/datasets/CKD_preprocessed.csv")

# Define X and y
X = df.drop('classification', axis=1)
y = df['classification']

models = {
    # 'rf': ['random_forest_model.pkl', 42],
    'knn': ["knn_model.pkl", 4786],
    # 'dt': ["decision_tree_model.pkl", 9],
    'svm': ["svm_model.pkl", 42],
    'gnb': ["gnb_model.pkl", 8077],
    'lr': ["lr_model.pkl", 42],
    'sgd': ["sgd_model.pkl", 199],
    # 'xt': ["xt_model.pkl", 199]
}

for key, value in models.items():
    print(key, "scores: ")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=value[1])
    loaded_model = joblib.load("mlscripts/models/"+value[0])
    y_pred = loaded_model.predict(X_test)
    print(y_pred)
    print(loaded_model.score(X_test, y_test))

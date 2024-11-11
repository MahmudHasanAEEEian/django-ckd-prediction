from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
import missingno as msno
import lime
from lime import lime_tabular

class Command(BaseCommand):
    help = 'Train models and evaluate performance'

    def handle(self, *args, **kwargs):
        # Load and preprocess the data
        df = pd.read_csv("path_to_your_data/kidney_disease.csv", index_col=0)
        df = df.dropna()
        
        # Data Cleaning
        df['wc'] = df['wc'].astype(float, errors='ignore')
        df['pcv'] = df['pcv'].astype(float, errors='ignore')
        df['rc'] = df['rc'].astype(float, errors='ignore')
        df['classification'] = df['classification'].apply(lambda x: 1 if 'ckd' in x else 0).astype(float)
        
        # Feature Selection
        df_fr_f_imprtance = df.drop(df.columns[[2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23]], axis=1)
        importances = mutual_info_classif(df_fr_f_imprtance.drop('classification', axis=1), df_fr_f_imprtance.classification, random_state=42)
        
        # Feature Processing
        X_num = df.drop(['age', 'bp', 'pot', 'wc', 'classification', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], axis=1)
        X_cat = df.drop(['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'classification'], axis=1)
        from sklearn.preprocessing import LabelEncoder
        X_cat1 = df[X_cat.drop('sg', axis=1).columns.tolist()].apply(LabelEncoder().fit_transform)
        final_df = pd.concat([X_num, X_cat.sg, X_cat1, df.classification], axis=1, join='inner')
        
        X = final_df.drop('classification', axis=1)
        y = final_df.classification
        
        # Model Training and Evaluation
        def train_and_evaluate(model, X_train, X_test, y_train, y_test):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print(model.__class__.__name__)
            print("Training Accuracy:", model.score(X_train, y_train))
            print("Test Accuracy:", model.score(X_test, y_test))
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))
            plt.figure(figsize=(10,7))
            sn.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='YlGnBu')
            plt.xlabel('Predicted')
            plt.ylabel('Truth')
            plt.show()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
        # Example models
        models = [
            RandomForestClassifier(random_state=1),
            KNeighborsClassifier(n_neighbors=1),
            DecisionTreeClassifier(random_state=2),
            SVC(kernel='linear', probability=True, random_state=0),
            GaussianNB(),
            LogisticRegression(random_state=0),
            SGDClassifier(loss='log', random_state=192),
            ExtraTreesClassifier(random_state=190)
        ]
        
        for model in models:
            train_and_evaluate(model, X_train, X_test, y_train, y_test)

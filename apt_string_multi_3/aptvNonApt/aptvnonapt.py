import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from collections import Counter
import math
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Disable GPU for TensorFlow
tf.config.set_visible_devices([], 'GPU')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Final_Datasets_just_string_Balance.csv')

# Filter the dataset to include all APTs (1), and combine legacy malware (2) and benign (3)
df_apt = df[df['TARGET'] == 1]  # Select APT rows
df_non_apt = df[df['TARGET'].isin([2, 3])]  # Select legacy malware and benign rows

# Check the number of samples
print(f"APT samples: {len(df_apt)}")
print(f"Non-APT (legacy malware + benign) samples: {len(df_non_apt)}")

# Randomly sample an equal number of rows from Non-APT to match APT (2005 samples)
df_non_apt_sampled = df_non_apt.sample(n=2005, random_state=42)

# Combine APT and the sampled Non-APT rows
df_balanced = pd.concat([df_apt, df_non_apt_sampled])

# Transform target variable: APT (1) remains as is, Non-APT (legacy malware = 2 and benign = 3) will be combined and set to 0
df_balanced['TARGET'] = df_balanced['TARGET'].replace({2: 0, 3: 0})

# Feature extraction: N-Grams and Entropy
def calculate_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

# Prepare features and target
X = df_balanced['STRING']
y = df_balanced['TARGET']

# TF-IDF Vectorization
print("Performing TF-IDF vectorization...")
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
X_tfidf = vectorizer.fit_transform(X)

# Add entropy as a feature
print("Calculating entropy...")
X_entropy = np.array([calculate_entropy(str(x)) for x in X]).reshape(-1, 1)
X_combined = np.hstack((X_tfidf.toarray(), X_entropy))

del X_tfidf, X_entropy
gc.collect()

# Feature selection
print("Performing feature selection...")
selector = SelectKBest(chi2, k=5000)
X_selected = selector.fit_transform(X_combined, y)

del X_combined
gc.collect()

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

del X_selected, y_encoded
gc.collect()

# Define machine learning models and hyperparameters
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), {
        'C': uniform(0.1, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }),
    'Random Forest': (RandomForestClassifier(), {
        'n_estimators': randint(100, 300),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }),
    'Gradient Boosting': (GradientBoostingClassifier(), {
        'n_estimators': randint(100, 300),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }),
    'k-NN': (KNeighborsClassifier(), {
        'n_neighbors': randint(3, 10),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }),
    'Gaussian Naive Bayes': (GaussianNB(), {}),
    'SVC': (SVC(probability=True), {
        'C': uniform(0.1, 10),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': uniform(0.01, 0.5)
    }),
    'AdaBoost': (AdaBoostClassifier(), {
        'n_estimators': randint(50, 150),
        'learning_rate': uniform(0.01, 0.5)
    }),
    'Linear Discriminant Analysis': (LinearDiscriminantAnalysis(), {}),
    'Quadratic Discriminant Analysis': (QuadraticDiscriminantAnalysis(), {}),
    'Multinomial Naive Bayes': (MultinomialNB(), {
        'alpha': uniform(0.1, 1)
    })
}

# Function to train and evaluate a model
def train_and_evaluate(name, model, params):
    try:
        print(f"Training {name}...")
        search = RandomizedSearchCV(model, params, n_iter=10, cv=3, n_jobs=1, random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Perform cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='accuracy')
        
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_scores': cv_scores,
            'confusion_matrix': conf_matrix
        }
    except Exception as e:
        print(f"Error in {name}: {str(e)}")
        return None

# Train and evaluate models
print("Training models...")
results = [train_and_evaluate(name, model, params) for name, (model, params) in models.items()]

# Filter out failed models
results = [r for r in results if r is not None]

# Print out results
for result in results:
    print(f"\nModel: {result['name']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1']:.4f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])

print("\nScript completed successfully!")


import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import gc
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
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

# Set memory limit for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the dataset in chunks
chunk_size = 10000  # Adjust based on your system's memory
df_chunks = pd.read_csv('Final_Datasets_just_string_Balance.csv', chunksize=chunk_size)

# Feature extraction: N-Grams and Entropy
def calculate_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

# Process data in chunks
X_list = []
y_list = []

for chunk in df_chunks:
    X_list.append(chunk['STRING'])
    y_list.append(chunk['TARGET'])

X = pd.concat(X_list)
y = pd.concat(y_list)

del X_list, y_list
gc.collect()

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)  # Reduced max_features
X_tfidf = vectorizer.fit_transform(X)

# Add entropy as a feature
X_entropy = np.array([calculate_entropy(str(x)) for x in X]).reshape(-1, 1)
X_combined = np.hstack((X_tfidf.toarray(), X_entropy))

del X_tfidf, X_entropy
gc.collect()

# Feature selection
selector = SelectKBest(chi2, k=2500)  # Reduced number of features
X_selected = selector.fit_transform(X_combined, y)

del X_combined
gc.collect()

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

del X_selected, y_encoded
gc.collect()

# Function to create a Keras model
def create_model(optimizer='adam', neurons=64, dropout_rate=0.3):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(neurons // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define models with hyperparameter search spaces
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), {
        'C': uniform(0.1, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }),
    'Random Forest': (RandomForestClassifier(), {
        'n_estimators': randint(50, 200),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }),
    'Gradient Boosting': (GradientBoostingClassifier(), {
        'n_estimators': randint(50, 200),
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
    'Neural Network': (KerasClassifier(build_fn=create_model, epochs=30, batch_size=64, verbose=0), {
        'optimizer': ['adam', 'rmsprop'],
        'neurons': randint(32, 96),
        'dropout_rate': uniform(0.1, 0.4)
    }),
    'SVC': (SVC(probability=True), {
        'C': uniform(0.1, 10),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': uniform(0.01, 0.5)
    }),
    'AdaBoost': (AdaBoostClassifier(), {
        'n_estimators': randint(30, 100),
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
        search = RandomizedSearchCV(model, params, n_iter=5, cv=3, n_jobs=1, random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Perform cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='accuracy')
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'best_params': search.best_params_,
            'cv_scores': cv_scores,
            'confusion_matrix': conf_matrix
        }
    except Exception as e:
        print(f"Error in {name}: {str(e)}")
        return None

# Train and evaluate models sequentially
results = []
for name, (model, params) in models.items():
    result = train_and_evaluate(name, model, params)
    if result:
        results.append(result)
    gc.collect()

# Prepare results for the table
table_data = []
for result in results:
    table_data.append([
        result['name'],
        f"{result['cv_scores'].mean():.4f} ± {result['cv_scores'].std():.4f}",
        f"{result['precision']:.4f} ± {result['cv_scores'].std():.4f}",
        f"{result['recall']:.4f} ± {result['cv_scores'].std():.4f}",
        f"{result['f1']:.4f} ± {result['cv_scores'].std():.4f}"
    ])

# Create and display the table
table = pd.DataFrame(table_data, columns=['Model', 'Accuracy (Mean ± Std)', 'Precision (Mean ± Std)', 'Recall (Mean ± Std)', 'F1 Score (Mean ± Std)'])
print(table.to_string(index=False))

# Plot confusion matrices
for result in results:
    plt.figure(figsize=(8, 6))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {result["name"]}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{result["name"].replace(" ", "_").lower()}.png')
    plt.close()

print("\nConfusion matrices have been saved as PNG files.")

# Plot ROC curves
plt.figure(figsize=(10, 8))
for result in results:
    if result['name'] != 'Neural Network':  # Skip Neural Network as it uses a different predict_proba method
        model = models[result['name']][0]
        model.set_params(**result['best_params'])
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y_train))):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[2], tpr[2], label=f'{result["name"]} (AUC = {roc_auc[2]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
plt.close()

# Save the best model
best_model = max(results, key=lambda x: x['accuracy'])
print(f"\nBest Model: {best_model['name']}")
print(f"Accuracy: {best_model['accuracy']:.4f}")

import joblib
joblib.dump(models[best_model['name']][0].set_params(**best_model['best_params']), 'best_model.joblib')

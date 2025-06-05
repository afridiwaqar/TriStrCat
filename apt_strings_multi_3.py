import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from collections import Counter
import math
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Disable GPU for TensorFlow (remove this if you have a compatible GPU)
tf.config.set_visible_devices([], 'GPU')

# Load the dataset
df = pd.read_csv('Final_Datasets_just_string_Balance.csv')

# Feature extraction: N-Grams and Entropy
def calculate_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

# Prepare features and target
X = df['STRING']
y = df['TARGET']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
X_tfidf = vectorizer.fit_transform(X)

# Add entropy as a feature
X_entropy = np.array([calculate_entropy(str(x)) for x in X]).reshape(-1, 1)
X_combined = np.hstack((X_tfidf.toarray(), X_entropy))

# Feature selection
selector = SelectKBest(chi2, k=5000)
X_selected = selector.fit_transform(X_combined, y)

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Define models with hyperparameter search spaces
models = {
    'Random Forest': (RandomForestClassifier(), {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }),
    'Gradient Boosting': (GradientBoostingClassifier(), {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }),
    'SVM': (SVC(probability=True), {
        'C': uniform(0.1, 10),
        'kernel': ['rbf', 'poly'],
        'gamma': uniform(0.01, 1)
    }),
    'Logistic Regression': (LogisticRegression(max_iter=1000), {
        'C': uniform(0.1, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    })
}

# Function to create a Keras model
def create_model(optimizer='adam', neurons=64, dropout_rate=0.3):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(neurons // 2, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Add Keras model to the list
models['Neural Network'] = (KerasClassifier(build_fn=create_model, epochs=50, batch_size=32, verbose=0), {
    'optimizer': ['adam', 'rmsprop'],
    'neurons': randint(32, 128),
    'dropout_rate': uniform(0.1, 0.5)
})

# Function to train and evaluate a model
def train_and_evaluate(name, model, params):
    print(f"Training {name}...")
    search = RandomizedSearchCV(model, params, n_iter=10, cv=3, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    return {
        'name': name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'best_params': search.best_params_
    }

# Train and evaluate models in parallel
results = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(name, model, params) for name, (model, params) in models.items())

# Print results
for result in results:
    print(f"\n{result['name']} Results:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-Score: {result['f1']:.4f}")
    print(f"ROC AUC: {result['roc_auc']:.4f}")
    print(f"Best Parameters: {result['best_params']}")

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
        for i in range(3):
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

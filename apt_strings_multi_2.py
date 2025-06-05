import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Importing TensorFlow/Keras for Neural Network
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dataset
df = pd.read_csv('Final_Datasets_just_string_Balance.csv')

# Feature extraction: N-Grams and Entropy
def calculate_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def generate_n_grams(text, n=3):
    n_grams = [text[i:i+n] for i in range(len(text)-n+1)]
    return ' '.join(n_grams)

df['N_GRAMS'] = df['STRING'].apply(lambda x: generate_n_grams(str(x), 3))
df['ENTROPY'] = df['STRING'].apply(lambda x: calculate_entropy(str(x)))

# Prepare features (N-grams and entropy) and target
X_ngrams = df['N_GRAMS']
X_entropy = df['ENTROPY'].values.reshape(-1, 1)
y = df['TARGET']

# Convert N-grams to vectorized format using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_ngrams_tfidf = vectorizer.fit_transform(X_ngrams).toarray()

# Concatenate N-grams and entropy as features
X = np.hstack((X_ngrams_tfidf, X_entropy))

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))

# Define models
models_dict = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'k-NN': KNeighborsClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Neural Network': None,  # Placeholder for Neural Network that we'll define below
    'SVC (Linear Kernel)': SVC(kernel='linear', probability=True),
    'SVC (RBF Kernel)': SVC(kernel='rbf', probability=True),
    'SVC (Poly Kernel)': SVC(kernel='poly', probability=True),
    'SVC (Sigmoid Kernel)': SVC(kernel='sigmoid', probability=True),
    'AdaBoost': AdaBoostClassifier(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Multinomial Naive Bayes': MultinomialNB(),
}

# Setup Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define Neural Network model (using Keras from TensorFlow)
def create_neural_network(input_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=input_dim))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Replace the None placeholder with the actual neural network model
models_dict['Neural Network'] = create_neural_network(X.shape[1])

# Lists to store results
results = {}
roc_figures = {}

# Loop through models
for model_name, model in models_dict.items():
    accuracy_scores, precision_scores, recall_scores, f1_scores, roc_auc_scores = [], [], [], [], []
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_index, test_index in skf.split(X, y_encoded):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Train model
        if model_name == 'Neural Network':  # Custom NN model
            model = create_neural_network(X_train.shape[1])  # Re-initialize NN model for each fold
            history = model.fit(X_train, pd.get_dummies(y_train), epochs=10, batch_size=32, validation_data=(X_test, pd.get_dummies(y_test)), class_weight=class_weights_dict, verbose=0)
            y_pred_prob = model.predict(X_test)
            y_pred = np.argmax(y_pred_prob, axis=1)
        else:
            model.fit(X_train, y_train)
            y_pred_prob = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        roc_auc_scores.append(roc_auc_score(y_test, y_pred_prob, multi_class='ovr'))
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(pd.get_dummies(y_test).to_numpy().ravel(), y_pred_prob.ravel())
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    
    # Calculate mean and standard deviation for scores
    results[model_name] = {
        'accuracy': (np.mean(accuracy_scores), np.std(accuracy_scores)),
        'precision': (np.mean(precision_scores), np.std(precision_scores)),
        'recall': (np.mean(recall_scores), np.std(recall_scores)),
        'f1': (np.mean(f1_scores), np.std(f1_scores)),
        'roc_auc': (np.mean(roc_auc_scores), np.std(roc_auc_scores)),
    }

    # Generate ROC curves
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'{model_name} (AUC = {mean_auc:.2f})', lw=2, alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_figures[model_name] = plt.gcf()
    plt.close()

# Display results like in the table format you provided
for model_name, scores in results.items():
    print(f"{model_name} - Accuracy: {scores['accuracy'][0]:.4f} ± {scores['accuracy'][1]:.4f}")
    print(f"{model_name} - Precision: {scores['precision'][0]:.4f} ± {scores['precision'][1]:.4f}")
    print(f"{model_name} - Recall: {scores['recall'][0]:.4f} ± {scores['recall'][1]:.4f}")
    print(f"{model_name} - F1-Score: {scores['f1'][0]:.4f} ± {scores['f1'][1]:.4f}")
    print(f"{model_name} - ROC AUC: {scores['roc_auc'][0]:.4f} ± {scores['roc_auc'][1]:.4f}")
    print("\n")

# To save the ROC figures
for model_name, fig in roc_figures.items():
    fig.savefig(f'{model_name}_roc_curve.png')


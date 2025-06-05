import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ML Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to compute entropy
def calculate_entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

# Function to generate N-grams
def generate_n_grams(text, n=3):
    n_grams = [text[i:i+n] for i in range(len(text)-n+1)]
    return ' '.join(n_grams)

# Load the CSV dataset
dataset_file = 'Final_Datasets_just_string_Balance.csv'  # Update this with your actual dataset path
df = pd.read_csv(dataset_file)

# Feature extraction: N-Grams and Entropy
df['N_GRAMS'] = df['STRING'].apply(lambda x: generate_n_grams(str(x), 3))  # 3-grams
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
y_encoded = LabelEncoder().fit_transform(y)  # Converts TARGET to 0, 1, 2

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Compute class weights for handling imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))

# 1. Neural Network (already in use)
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(layers.Dropout(0.3))  # Dropout to prevent overfitting
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # Multi-class output (3 classes)

# Compile the model using categorical crossentropy for multi-class classification
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights_dict)

# Evaluate the Neural Network model
nn_test_loss, nn_test_accuracy = model.evaluate(X_test, y_test)
print(f"Neural Network Test Accuracy: {nn_test_accuracy * 100:.2f}%")

# 2. Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
print(f"Logistic Regression Test Accuracy: {accuracy_score(y_test, y_pred_logreg) * 100:.2f}%")
print(classification_report(y_test, y_pred_logreg))

# 3. Random Forest
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print(classification_report(y_test, y_pred_rf))

# 4. k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(f"k-NN Test Accuracy: {accuracy_score(y_test, y_pred_knn) * 100:.2f}%")
print(classification_report(y_test, y_pred_knn))

# 5. Support Vector Machine
svc = SVC(class_weight='balanced')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print(f"SVM Test Accuracy: {accuracy_score(y_test, y_pred_svc) * 100:.2f}%")
print(classification_report(y_test, y_pred_svc))

# 6. Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(f"Naive Bayes Test Accuracy: {accuracy_score(y_test, y_pred_nb) * 100:.2f}%")
print(classification_report(y_test, y_pred_nb))

# Save the neural network model
model.save('apt_detection_model.keras')


import pandas as pd
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

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
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_ngrams_tfidf = vectorizer.fit_transform(X_ngrams).toarray()

# Concatenate N-grams and entropy as features
X = np.hstack((X_ngrams_tfidf, X_entropy))

# Encode target labels (ensure the labels are 1, 2, 3)
#label_encoder = LabelEncoder()
#y_encoded = label_encoder.fit_transform(y)  # 1 -> 1, 2 -> 2, 3 -> 3

y_encoded = pd.get_dummies(y).values
# One-hot encode the target labels
#y_encoded = pd.get_dummies(y_encoded + 1).values  # Ensure the labels are 1, 2, 3

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Compute class weights for handling imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(df['TARGET']), y=df['TARGET'])
class_weights_dict = dict(enumerate(class_weights))

# Build the neural network model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(layers.Dropout(0.3))  # Dropout to prevent overfitting
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # Multi-class output (3 classes)

# Compile the model using categorical crossentropy for multi-class classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weights_dict)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

label_map = {0: 1, 1: 2, 2: 3}
y_true_labels_mapped = np.vectorize(label_map.get)(y_true_labels)
y_pred_labels_mapped = np.vectorize(label_map.get)(y_pred_labels)

# Generate classification report
print(classification_report(y_true_labels_mapped, y_pred_labels_mapped))

# Save the model
model.save('apt_detection_model.keras')


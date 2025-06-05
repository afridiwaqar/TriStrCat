import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

# Assuming df is your DataFrame with the 'STRING' and 'TARGET' columns
# Replace 'YourDataset.csv' with the actual file path or URL of your dataset
df = pd.read_csv('Final_Datasets_just_string_Balance.csv')

# Data Cleaning (if needed)
df.dropna(inplace=True)

# Feature Extraction and Text Processing
# Encode special characters with unique numbers
special_char_encoding = {'!': 1, '@': 2, '#': 3, '$': 4, '%': 5, '&': 6, ';': 7, '*': 8, '^': 9, '<': 10}
df['STRING_encoded'] = df['STRING'].apply(lambda x: ''.join([str(special_char_encoding[char]) if char in special_char_encoding else char for char in x]))

# Model Selection, Hyperparameter Tuning, Cross-Validation, and Evaluation
X = df.drop(['TARGET'], axis=1)  # Features
y = df['TARGET']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor for one-hot encoding of categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['STRING_encoded']),
        # Add more transformers for other columns if needed
    ],
    remainder='passthrough'
)

# Define a pipeline with a RandomForestClassifier as an example model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Define hyperparameters for tuning
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
}

# Use GridSearchCV for hyperparameter tuning and cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model from hyperparameter tuning
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

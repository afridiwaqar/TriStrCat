import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pycaret.classification import *
from joblib import Parallel, delayed
from tqdm import tqdm

df = pd.read_csv('Final_Datasets_just_string_Balance.csv')

df = df.drop_duplicates()
df = df.fillna('missing')

le = LabelEncoder()
df['TARGET'] = le.fit_transform(df['TARGET'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['STRING'])

pipe = Pipeline([
 ('rfc', RandomForestClassifier())
])

param_grid = {
 'rfc__n_estimators': [100, 200, 300],
 'rfc__max_depth': [None, 10, 20, 30],
 'rfc__min_samples_split': [2, 5, 10]
}

# Define a function to fit a model with given parameters
def fit_model(params):
 model = pipe.set_params(**params)
 model.fit(X, df['TARGET'])
 return model

# List of parameters for each model
param_list = list(ParameterGrid(param_grid))

# Use joblib to fit models in parallel
models = Parallel(n_jobs=-1)(delayed(fit_model)(params) for params in tqdm(param_list))

# Now, you can evaluate the models, for example by calculating their scores
scores = [model.score(X, df['TARGET']) for model in models]

print("Scores: ", scores)

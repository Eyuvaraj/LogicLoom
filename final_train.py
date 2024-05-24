# *** GENERATED PIPELINE ***

# LOAD DATA
import pandas as pd
train_dataset = pd.read_csv("./train_set.csv", encoding="UTF-8", delimiter=",")

# DROP IGNORED COLUMNS
ignore_columns = ['Article_ID']
train_dataset = train_dataset.drop(ignore_columns, axis=1, errors="ignore")

import pickle


# PREPROCESSING-1
import re
import string
import nltk
TEXT_COLUMNS = ['Article_content']
def process_text(__dataset):
    for _col in TEXT_COLUMNS:
        process_text = [t.lower() for t in __dataset[_col]]
        # strip all punctuation
        table = str.maketrans('', '', string.punctuation)
        process_text = [t.translate(table) for t in process_text]
        # convert all numbers in text to 'num'
        process_text = [re.sub(r'\d+', 'num', t) for t in process_text]
        __dataset[_col] = process_text
    return __dataset
train_dataset = process_text(train_dataset)

# DETACH TARGET
TARGET_COLUMNS = ['Article_type']
feature_train = train_dataset.drop(TARGET_COLUMNS, axis=1)
target_train = train_dataset[TARGET_COLUMNS].copy()

# PREPROCESSING-2
from sklearn.feature_extraction.text import TfidfVectorizer
TEXT_COLUMNS = ['Article_content']
temp_train_data = feature_train[TEXT_COLUMNS]
# Make the entire dataframe sparse to avoid it converting into a dense matrix.
feature_train = feature_train.drop(TEXT_COLUMNS, axis=1).astype(pd.SparseDtype('float64', 0))
vectorizers = {}
for _col in TEXT_COLUMNS:
    tfidfvectorizer = TfidfVectorizer(max_features=3000)
    vector_train = tfidfvectorizer.fit_transform(temp_train_data[_col])
    feature_names = ['_'.join([_col, name]) for name in tfidfvectorizer.get_feature_names_out()]
    vector_train = pd.DataFrame.sparse.from_spmatrix(vector_train, columns=feature_names, index=temp_train_data.index)
    feature_train = pd.concat([feature_train, vector_train], axis=1)
    vectorizers[_col] = tfidfvectorizer
with open('tfidfVectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizers, f)

# BEST PARAMETERS IN THE CANDIDATE SCRIPT
# PLEASE SEE THE CANDIDATE SCRIPTS FOR THE HYPERPARAMTER OPTIMIZATION CODE
best_params = {'n_estimators': 154, 'algorithm': 'SAMME'}

# MODEL
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
random_state_model = 42
model = AdaBoostClassifier(random_state=random_state_model, **best_params)
model.fit(feature_train, target_train.values.ravel())
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

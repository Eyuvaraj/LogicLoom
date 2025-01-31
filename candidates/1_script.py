# *** GENERATED PIPELINE ***

# LOAD DATA
import pandas as pd
train_dataset = pd.read_csv("./train_set.csv", encoding="UTF-8", delimiter=",")

# TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
def split_dataset(dataset, train_size=0.85, random_state=1024):
    train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, random_state=random_state)
    return train_dataset, test_dataset	
train_dataset, test_dataset = split_dataset(train_dataset)
train_dataset, validation_dataset = split_dataset(train_dataset)

# SUBSAMPLE
# If the number of rows of train_dataset is larger than sample_size, sample rows to sample_size for speedup.
from lib.sample_dataset import sample_dataset
train_dataset = sample_dataset(
    dataframe=train_dataset,
    sample_size=100000,
    target_columns=['Article_type'],
    task_type='classification'
)

# DROP IGNORED COLUMNS
ignore_columns = ['Article_ID']
train_dataset = train_dataset.drop(ignore_columns, axis=1, errors="ignore")
validation_dataset = validation_dataset.drop(ignore_columns, axis=1, errors="ignore")

test_dataset = validation_dataset


# PREPROCESSING-1
# Component: Preprocess:TextPreprocessing
# Efficient Cause: Preprocess:TextPreprocessing is required in this pipeline since the dataset has ['feature:str_text_presence']. The relevant features are: ['Article_content'].
# Purpose: Preprocess and normalize text.
# Form:
#   Input: array of strings
#   Key hyperparameters used: None
# Alternatives: Although  can also be used for this dataset, Preprocess:TextPreprocessing is used because it has more  than .
# Order: Preprocess:TextPreprocessing should be applied  
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
test_dataset = process_text(test_dataset)

# DETACH TARGET
TARGET_COLUMNS = ['Article_type']
feature_train = train_dataset.drop(TARGET_COLUMNS, axis=1)
target_train = train_dataset[TARGET_COLUMNS].copy()
feature_test = test_dataset.drop(TARGET_COLUMNS, axis=1)
target_test = test_dataset[TARGET_COLUMNS].copy()

# PREPROCESSING-2
# Component: Preprocess:TfidfVectorizer
# Efficient Cause: Preprocess:TfidfVectorizer is required in this pipeline since the dataset has ['feature:str_text_presence']. The relevant features are: ['Article_content'].
# Purpose: Convert a collection of raw documents to a matrix of TF-IDF features.
# Form:
#   Input: raw_documents
#   Key hyperparameters used: 
#		 "max_features: int, default=None" :: If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. This parameter is ignored if vocabulary is not None.
# Alternatives: Although  can also be used for this dataset, Preprocess:TfidfVectorizer is used because it has more  than .
# Order: Preprocess:TfidfVectorizer should be applied  
from sklearn.feature_extraction.text import TfidfVectorizer
TEXT_COLUMNS = ['Article_content']
temp_train_data = feature_train[TEXT_COLUMNS]
temp_test_data = feature_test[TEXT_COLUMNS]
# Make the entire dataframe sparse to avoid it converting into a dense matrix.
feature_train = feature_train.drop(TEXT_COLUMNS, axis=1).astype(pd.SparseDtype('float64', 0))
feature_test = feature_test.drop(TEXT_COLUMNS, axis=1).astype(pd.SparseDtype('float64', 0))
for _col in TEXT_COLUMNS:
    tfidfvectorizer = TfidfVectorizer(max_features=3000)
    vector_train = tfidfvectorizer.fit_transform(temp_train_data[_col])
    feature_names = ['_'.join([_col, name]) for name in tfidfvectorizer.get_feature_names_out()]
    vector_train = pd.DataFrame.sparse.from_spmatrix(vector_train, columns=feature_names, index=temp_train_data.index)
    feature_train = pd.concat([feature_train, vector_train], axis=1)
    vector_test = tfidfvectorizer.transform(temp_test_data[_col])
    vector_test = pd.DataFrame.sparse.from_spmatrix(vector_test, columns=feature_names, index=temp_test_data.index)
    feature_test = pd.concat([feature_test, vector_test], axis=1)

# MODEL
import numpy as np
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(feature_train, target_train.values.ravel())
y_pred = model.predict_proba(feature_test)
# POST PROCESSING
if np.shape(y_pred)[1] == 2:
    y_pred = y_pred[:, 1]

#EVALUATION
from sklearn.metrics import roc_auc_score
__roc_auc = roc_auc_score(target_test, y_pred)
print('RESULT: ROC AUC:', str(__roc_auc))

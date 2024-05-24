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

# HYPERPARAMETER OPTIMIZATION
import optuna
from sklearn.linear_model import LogisticRegression
# NEED CV: ex.) optuna.integration.OptunaSearchCV()
class Objective(object):
    def __init__(self, feature_train, target_train, feature_test, target_test, __random_state):
        self.feature_train = feature_train
        self.target_train = target_train
        self.feature_test = feature_test
        self.target_test = target_test 
        self.__random_state = __random_state
    def __call__(self, trial):
        def set_hyperparameters(trial):
            params = {}
            #params['penalty'] =  trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']) # l2 
            params['C'] = trial.suggest_loguniform('C', 1e-5, 1e2) # 1 
            #params['solver'] = trial.suggest_categorical('solver', ['lbfgs','saga']) # lbfgs 
            params['class_weight'] = trial.suggest_categorical('class_weight', ['balanced', None]) # None 
            return params
        
        # SET DATA
        import numpy as np
    
        if isinstance(self.feature_train, pd.DataFrame):
            feature_train = self.feature_train
        elif isinstance(self.feature_train, np.ndarray):
            feature_train = pd.DataFrame(self.feature_train)
        else:
            feature_train = pd.DataFrame(self.feature_train.toarray())
    
        if isinstance(self.target_train, pd.DataFrame):
            target_train = self.target_train
        elif isinstance(self.target_train, np.ndarray):
            target_train = pd.DataFrame(self.target_train)
        else:
            target_train = pd.DataFrame(self.target_train.toarray())
    
        if isinstance(self.feature_test, pd.DataFrame):
            feature_test = self.feature_test
        elif isinstance(self.feature_test, np.ndarray):
            feature_test = pd.DataFrame(self.feature_test)
        else:
            feature_test = pd.DataFrame(self.feature_test.toarray())
    
        if isinstance(self.target_test, pd.DataFrame):
            target_test = self.target_test
        elif isinstance(self.target_test, np.ndarray):
            target_test = pd.DataFrame(self.target_test)
        else:
            target_test = pd.DataFrame(self.target_test.toarray())
        # MODEL 
        params = set_hyperparameters(trial)
        model = LogisticRegression(random_state=self.__random_state, **params)
        model.fit(feature_train, target_train.values.ravel())
        y_pred = model.predict_proba(feature_test)
        # POST PROCESSING
        if np.shape(y_pred)[1] == 2:
            y_pred = y_pred[:, 1]
        
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(target_test, y_pred)
        
        return score
    
n_trials = 10
timeout = 120 
random_state = 42 
random_state_model = 42 
direction = 'maximize' 
    
study = optuna.create_study(direction=direction,
                sampler=optuna.samplers.TPESampler(seed=random_state)) 
default_hyperparameters = {'C': 1.0, 'class_weight': None, 'penalty': 'l2', 'solver': 'lbfgs'}
study.enqueue_trial(default_hyperparameters)
study.optimize(Objective(feature_train, target_train, feature_test, target_test, random_state_model), 
                n_trials=n_trials, 
                timeout=timeout)
best_params = study.best_params
print("best params:", best_params)
print("RESULT: ROC_AUC: " + str(study.best_value))

import pickle

import datasets
import yaml
import pandas as pd
import re
import nltk.corpus
import nltk
import joblib
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn import set_config
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=100, window=5, min_count=1, workers=4):
        self.max_features = max_features
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.word2vec = None

    def fit(self, X, y=None):
        sentences = [sentence.split() for sentence in X]
        self.word2vec = Word2Vec(sentences, vector_size=self.max_features, window=self.window,
                                 min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X, y=None):
        vectors = []
        for sentence in X:
            sentence_vector = []
            for word in sentence.split():
                if word in self.word2vec.wv.key_to_index:
                    sentence_vector.append(self.word2vec.wv[word])
            if sentence_vector:
                sentence_vector = np.mean(sentence_vector, axis=0)
                vectors.append(sentence_vector)
            else:
                vectors.append(np.zeros(self.max_features))
        return np.array(vectors)

def main():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    with open("data/X_train.pkl", "rb") as f:
        X_train = pickle.load(file=f)
    with open("data/X_test.pkl", "rb") as f:
        X_test = pickle.load(file=f)
    with open("data/y_train.pkl", "rb") as f:
        y_train = pickle.load(file=f)
    with open("data/y_test.pkl", "rb") as f:
        y_test = pickle.load(file=f)

    review_preprocessing = Pipeline([('bow_review', TfidfVectorizer())])
    summary_preprocessing = Pipeline([('bow_summary', TfidfVectorizer())])
    numeric_preprocessing = Pipeline([('scaler', MinMaxScaler())])

    preprocess = ColumnTransformer([
        ('review_preprocessing', review_preprocessing, 'reviewText'),
        ('summary_preprocessing', summary_preprocessing, 'summary'),
        ('numeric_preprocessing', numeric_preprocessing, ['ReviewTextLen', 'unixReviewTime'])])


    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('clf', RandomForestClassifier())])

    set_config(display='diagram')

    param_grid = [
        {
            # "preprocess__review_preprocessing__bow_review": [CountVectorizer(), TfidfVectorizer(),
            #  Word2VecTransformer()],
            # "preprocess__summary_preprocessing__bow_summary": [CountVectorizer(), TfidfVectorizer(),
            # Word2VecTransformer()],
            "preprocess__summary_preprocessing__bow_summary__max_features": [50, 100, 200, 500],
            "preprocess__review_preprocessing__bow_review__max_features": [50, 100, 200, 500]
        }]

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1,
                                        scoring='f1_macro')
    randomized_search = RandomizedSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1,
                                           scoring='f1_macro')
    halvingrandom_search = HalvingRandomSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1,
                                             scoring='f1_macro')
    halvinggrid_search = HalvingGridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=-1,
                                             scoring='f1_macro')



    #testing
    print("start :( ")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    print("grid done \n")
    ###
    start_time = time.time()
    randomized_search.fit(X_train, y_train)
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    print("randomized done \n")
    # havlingrandom
    start_time = time.time()
    halvingrandom_search.fit(X_train, y_train)
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    print("halving done- jeszcze troche nie poddawaj sie\n")
    # alvinggrid
    start_time = time.time()
    halvinggrid_search.fit(X_train, y_train)
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)


    # print(grid_search.best_params_)
    # print(randomized_search.best_params_)
    # print(halvingrandom_search.best_params_)
    # print(halvinggrid_search.best_params_)


    joblib.dump(grid_search.best_estimator_, 'data/grid_search_model.joblib')
    joblib.dump(randomized_search.best_estimator_, 'data/randomized_search_model.joblib')
    joblib.dump(halvingrandom_search.best_estimator_, 'data/halvingrandom_search_model.joblib')
    joblib.dump(halvinggrid_search.best_estimator_,  'data/halvinggrid_search_model.joblib')
    joblib.dump(pipeline, 'data/pipe.joblib')

main()
import pickle
import datasets
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import spacy
from spacy.tokens import Token
from spacy.matcher import Matcher
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_sm")
spacy.prefer_gpu()
nlp.add_pipe("spacytextblob")
def remove_stopwords(description):
    words = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", description)
    words = nltk.word_tokenize(words)
    stemmed_words = [stemmer.stem(word) for word in words]
    filtered_words = [word for word in stemmed_words if word.lower() not in stopwords]
    rev =  ' '.join(filtered_words)
    return rev
 # spacysentiment
def remove_stopwords_spacy(description):
    doc = nlp(description)
    filtered_text = ' '.join(token.text for token in doc if token._.blob.sentiment_assessments.assessments)
    return filtered_text

def main():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    with open("data/dataset.pkl", "rb") as f:
        df = pickle.load(file=f)

    # Pre-preprocessing
    df['overall'] = df['overall'].replace({1: 0, 2: 0, 3: 1, 4: 2, 5: 2})
    df['verified'] = df['verified'].replace({True: 1, False: 0})
    df["ReviewTextLen"] = df["reviewText"].astype(str).fillna('').apply(len)
    df = df.drop(columns=['image', 'vote', 'style'])
    df['summary'] = df['summary'].fillna(value=df['reviewText'])
    df['reviewText'] = df['reviewText'].fillna(value=df['summary'])
    df = df.drop(['reviewTime', 'reviewerID', 'asin', 'reviewerName'], axis=1)
    
    df['reviewText'] = df['reviewText'].astype(str).str.lower()
    df['summary'] = df['summary'].astype(str).str.lower()

    #### remove stopwords 
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words("english")

    df['reviewText'] = df['reviewText'].apply(remove_stopwords_spacy)
    df['summary'] = df['summary'].apply(remove_stopwords_spacy)


    X_train, X_test, y_train, y_test = train_test_split(df.drop("overall", axis=1),
                                                        df["overall"], stratify = df["overall"],  test_size=cfg['split_size'], random_state=42)
    with open("data/X_train.pkl", "wb") as f:
        pickle.dump(obj=X_train, file=f)
    with open("data/X_test.pkl", "wb") as f:
        pickle.dump(obj=X_test, file=f)

    with open("data/y_train.pkl", "wb") as f:
        pickle.dump(obj=y_train, file=f)

    with open("data/y_test.pkl", "wb") as f:
        pickle.dump(obj=y_test, file=f)


main()
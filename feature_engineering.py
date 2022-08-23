import pandas as pd
import numpy as np
import joblib
import yaml
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from data_processing import text_preprocess




def get_categories(data, path):
    """
    Description: Encode category column
    param data: dataset
    return: dataset
    """
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'Category_id'.
    data['Category_id'] = label_encoder.fit_transform(data['Category'])
    joblib.dump(label_encoder, path)
    return data


def text_norm(t, x, pt, px):
    """
    Descrption: vectorize train and test data using tfidf and dump for model training

    """
    X_train = t.fit_transform(x).toarray()
    joblib.dump(t, pt)
    #X_test = t.transform(y).toarray()
    joblib.dump(X_train, px)
    #joblib.dump(X_test, py)


def feature_engineering(params):
    # dataset_path
    train_path = params['data_source']['train']

    # tfidf parametrs
    max_features = params['tfidf']['max_features']
    max_df = params['tfidf']['max_df']
    norm = params['tfidf']['norm']
    encoding = params['tfidf']['encoding']
    ngram_range = params['tfidf']['ngram_range']
    stop_words = params['tfidf']['stop_words']

    tfidf = TfidfVectorizer(
        max_features=max_features,
        max_df=max_df,
        norm=norm,
        encoding=encoding,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )

    # save path for tfidf
    tfidf_path = params['load_data']['tfidf']
    # save path for encoder
    en_path = params['load_data']['label_encoder']
    # save path data after data split
    x_train_path = params['load_data']['X_train']
    x_test_path = params['load_data']['X_test']
    y_train_path = params['load_data']['y_train']
    y_test_path = params['load_data']['y_test']

    # import dataset
    train_data = pd.read_csv("data_processed/data_processed.csv")
    # Tokenization
    #train_data_norm = text_preprocess(train_data)
    train_data_norm = tfidf.fit_transform(train_data['Text_processed']).toarray()
    # get categories
    """y_trn = get_categories(train_data, en_path)

    # split data
    random_state = params['base']['random_state']
    X_train,X_test,y_train,y_test = train_test_split(train_data_norm,
                                                    y_trn['Category_id'],
                                                    test_size=0.20,
                                                    random_state=random_state)"""

    #X_train = tfidf.fit_transform(X_train).toarray()
    joblib.dump(tfidf, tfidf_path)
    # X_test = t.transform(y).toarray()
    #joblib.dump(X_train, x_train_path)
    #joblib.dump(X_test, x_test_path)

    # tfidf/ feature extraction of train and test data set
    #text_norm(tfidf, X_train, tfidf_path, x_train_path)


    # y_train, y_test
    #joblib.dump(y_train, y_train_path)
    #joblib.dump(y_test, y_test_path)


def get_parameter(path):
    """
    Description: loads all the parameter in yaml file
    param path:
    return:
    """
    with open(path, encoding='utf-8') as p:
        yaml_parameter = yaml.safe_load(p)
    return yaml_parameter



if __name__ == "__main__":
    path = "params.yaml"
    params = get_parameter(path)
    feature_engineering(params)
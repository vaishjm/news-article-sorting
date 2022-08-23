
import numpy as numpy
import pandas as pd
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
"""
# read the dataset
# plot the dataset
# save the plot of dataset"""

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")




def remove_special_characters(text):
    """
    Description: Removes <br> tags and any other characters except alphabets and apostrophe.
    Param path:
    Returns: str
    """
    text = re.sub("<br\\s*/?>", " ", text)
    text = re.sub("[^a-zA-Z']", " ", text)
    text = re.sub("-", " ", text)
    return text


def lemmatize(text):
    """
    Description: Lemmatizes input string. eg: Bats -> Bat , also check and not include it in dict if the word is english stopword
    Param path:
    Returns: str
    """

    text = text.split()
    wordnet = WordNetLemmatizer()
    text = [wordnet.lemmatize(word) for word in text if word not in set(stopwords('english'))]
    return " ".join(text)



def text_preprocess(data):
    """
    Description: performs processing on text data like cleaning and normalization
    param data: DataFrame
    return: DataFrame with 'Text_processed' column
    """
    combo = []
    for i in range(0, len(data)):
        text = remove_special_characters(data['Text'][i])
        text = text.lower()
        text = lemmatize(text)

        combo.append(text)
    data['Text_processed'] = pd.DataFrame(combo)
    return data


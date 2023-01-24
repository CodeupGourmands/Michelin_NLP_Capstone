import re
import unicodedata
from typing import List

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords as stpwrds
from sklearn.model_selection import train_test_split

stopwords = stpwrds.words('english')

EXTRA_WORDS = []
EXCLUDE_WORDS = []


def clean_michelin(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function was the basic clean used before scraping the data
    into the new data frame
    '''
    # Make all column names lowercase
    df.columns = df.columns.str.lower()
    # renaming columns in snake case
    df = df.rename(columns={"phonenumber": "phone_number",
                            "websiteurl": "website_url",
                            "facilitiesandservices": "facilities_and_services"}
                   )
    return df


def basic_clean(string_to_clean: str) -> str:
    '''
    Changes string to lowercase and removes any non-ASCII characters
    from a document string
    ## Parameters
    string_to_clean: string containing the document to be cleaned
    ## Returns
    cleaned string
    '''
    string_to_clean = string_to_clean.lower()
    string_to_clean = unicodedata.normalize('NFKD', string_to_clean).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    string_to_clean = re.sub(r"[^a-z0-9'\s]", '', string_to_clean)
    return string_to_clean


def tokenize(string_to_clean: str) -> str:
    '''
    Tokenizes given document string
    ## Parameters
    string_to_clean: string containing document to be tokenized
    ## Returns
    Tokenized document string
    '''
    tok = nltk.tokenize.ToktokTokenizer()
    return tok.tokenize(string_to_clean, return_str=True)


def stem(tokens: str) -> str:
    '''
    Generates stems of given document string
    ## Parameters
    tokens: tokenized document string
    ## Returns
    stemmed string
    '''
    ps = nltk.porter.PorterStemmer()
    ret = [ps.stem(s) for s in tokens.split()]
    return ' '.join(ret)


def lemmatize(tokens: str) -> str:
    '''
    Lemmatizes given document string
    ## Parameters
    tokens: tokenized document string
    ## Returns
    lemmatized string
    '''
    lem = nltk.stem.WordNetLemmatizer()
    ret = [lem.lemmatize(s) for s in tokens.split()]
    return ' '.join(ret)


def remove_stopwords(string_to_clean: str,
                     extra_words: List[str] = EXTRA_WORDS,
                     exclude_words: List[str] = EXCLUDE_WORDS) -> str:
    '''
    Removes stopwords from string
    ## Parameters
    string_to_clean: document string to be cleaned
 =a+extra_words: additional stop words to remove from `string_to_clean`
    exclude_words: stopwords to keep in `string_to_clean`
    ## Returns
    document string with stopwords removed
    '''
    string_to_clean = [t for t in string_to_clean.split()]
    for exc in exclude_words:
        stopwords.remove(exc)
    for ext in extra_words:
        stopwords.append(ext)
    stopped = [t for t in string_to_clean if t not in stopwords]
    return ' '.join(stopped)


def squeaky_clean(string_to_clean: str,
                  extra_words: List[str] = EXTRA_WORDS,
                  exclude_words: List[str] = EXCLUDE_WORDS) -> str:
    '''
    Performs basic cleaning, removes stopwords, and tokenizes given
    document string

    ## Parameters
    string_to_clean: document string to be cleaned
    extra_words: additional stopwords to remove from document string
    exclude_words: stopwords to keep in document string
    ## Returns
    cleaned string
    '''
    string_to_clean = basic_clean(string_to_clean)
    string_to_clean = tokenize(string_to_clean)
    return remove_stopwords(string_to_clean, extra_words, exclude_words)


def process_nl(document_series: pd.Series,
               extra_words: List[str] = [],
               exclude_words: List[str] = []) -> pd.DataFrame:
    '''
    cleans, stems, and lemmatizes given series of document strings
    ## Parameters
    document_series: `Series` containing document strings
    extra_words: additional stopwords to remove from document string
    exclude_words: stopwords to keep in document string
    ## Returns
    `DataFrame` containing the cleaned, stemmed, and lemmatized string
    '''
    ret_df = pd.DataFrame()
    ret_df['clean'] = document_series.apply(
        squeaky_clean, exclude_words=exclude_words, extra_words=extra_words)
    ret_df['lemmatized'] = ret_df['clean'].apply(lemmatize)
    return ret_df

def tvt_split(df: pd.DataFrame,
              stratify: str = None,
              tv_split: float = .2,
              validate_split: int = .3):
    '''tvt_split takes a pandas DataFrame,
    a string specifying the variable to stratify over,
    as well as 2 floats where 0 < f < 1 and
    returns a train, validate, and test split of the DataFame,
    split by tv_split initially and validate_split thereafter. '''
    strat = df[stratify]
    train_validate, test = train_test_split(
        df, test_size=tv_split, random_state=911, stratify=strat)
    strat = train_validate[stratify]
    train, validate = train_test_split(
        train_validate, test_size=validate_split,
        random_state=911, stratify=strat)
    return train, validate, test

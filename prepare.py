import re
import unicodedata
from typing import List, Union, Tuple

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from nltk.corpus import stopwords as stpwrds
from sklearn.model_selection import train_test_split

# TODO Woody organize based on execution order
EXTRA_WORDS: List[str] = ['dishes', 'restaurant', 'dining', 'chef',
                          'menu', 'cuisine', 'there',
                          'ingredients', 'flavour', 'also',
                          'dish', 'ingredient']
EXCLUDE_WORDS: List[str] = []

NGRAMS_TO_REMOVE: List[str] = ['update september 2020', 'last update september',
                               'wine list']

stopwords = stpwrds.words('english') + EXTRA_WORDS


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
    string_to_clean = re.sub(r"[^a-z0-9\s]", '', string_to_clean)
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
                     extra_words: List[str],
                     exclude_words: List[str]) -> str:
    '''
    Removes stopwords from string
    ## Parameters
    string_to_clean: document string to be cleaned
    extra_words: additional stop words to remove from `string_to_clean`
    exclude_words: stopwords to keep in `string_to_clean`
    ## Returns
    document string with stopwords removed
    '''
    stopped = [word for word in string_to_clean.split()
               if word not in stopwords]
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
    string_to_clean = remove_stopwords(
        string_to_clean, extra_words, exclude_words)
    string_to_clean = remove_ngrams(string_to_clean, NGRAMS_TO_REMOVE)
    return string_to_clean


def process_nl(document_series: pd.Series,
               extra_words: List[str] = EXTRA_WORDS,
               exclude_words: List[str] = EXCLUDE_WORDS) -> pd.DataFrame:
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
        squeaky_clean,
        exclude_words=exclude_words,
        extra_words=extra_words).astype('string')
    ret_df['lemmatized'] = ret_df['clean'].apply(lemmatize).astype('string')
    return ret_df


def change_dtype_str(df: pd.DataFrame) -> pd.DataFrame:
    '''
    ## Description:
    This is a custom Function to change dtype to string
        as appropraiate for this project.
    ## Parameters:
    df = `DataFrame` containing michelin data
    ## Returns:
    `DataFrame` with dtypes changed as appropriate
    '''
    df.name = df.name.fillna('').astype('string')
    df.address = df.address.fillna('').astype('string')
    df.location = df.location.fillna('').astype('string')
    df.cuisine = df.cuisine.fillna('').astype('string')
    df.facilities_and_services = df.facilities_and_services.fillna(
        'NONE').astype('string')
    df.award = df.award.fillna('').astype('category')
    df.data = df.data.fillna('').astype('string')
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes in the dataframe, drops unnecessary columns,
    and creates new columns/features for exploration and potential
    classification purposes. It returns the dataframe with additional
    columns created.
    '''
    # Dropping restaurants no longer listed in guide
    df = df[df.data != 'None']

    # Dropping unnecessary columns
    df = df.drop(['phone_number', 'website_url'], axis=1)

    # Lower case all column values if column is object/string type
    df = df.apply(lambda x: x.str.lower() if (x.dtype == 'object') else x)

    # Turn NaN values in price to 'nothing', so that it can be recast
    df['price'] = df['price'].fillna('').astype('str')
    # Casting a new column, price level, using length of column
    df['price_level'] = df['price'].apply(lambda x: len(x))
    # impute price level "0" with the mode for this column
    mode = df.price_level.mode()[0]
    df['price_level'] = df['price_level'].replace(0, mode)

    # splitting location columns into two columns
    df[['city', 'country']] = df['location'].str.split(', ', 1, expand=True)
    # Turn cities into city-states, impute these into country column
    df['country'] = np.where(pd.isna(df['country']), df['city'], df['country'])

    return df


def sentiment_score(lemmatized: str, sia: SentimentIntensityAnalyzer) -> float:
    # TODO Yuvia docstring
    return sia.polarity_scores(lemmatized)['compound']


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


def prepare_michelin(df: pd.DataFrame,
                     split: bool = True) -> Union[pd.DataFrame,
                                                  Tuple[pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.DataFrame]]:
    '''
    Prepares Michelin DataFrame
    ## Parameters
    df: `DataFrame` with Michelin data
    split: Boolean for whether or not to split the data, default True
    ## Returns
    either a DataFrame or a tuple of the Train, Validate, and Test
    `DataFrame`
    '''
    df = create_features(df)
    df = change_dtype_str(df)
    lemmatized = process_nl(df.data)
    df = pd.concat([df, lemmatized], axis=1)
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df.lemmatized.apply(sentiment_score, sia=sia)
    df['word_count'] = df.lemmatized.str.split().apply(len)
    if split:
        return tvt_split(df, stratify='award')
    return df

# TODO Cristina either move to a new file or delete, as it's not used in the workbook


def prep_classification_data(train, validate, test):
    '''
    This function takes in train, validate, and test and returns
    train, validate, and test prepped for classification modeling
    '''
    # Create dummy columns
    dummy_train = pd.get_dummies(
        train, columns=['country', 'price_level'], drop_first=False)
    dummy_validate = pd.get_dummies(
        validate, columns=['country', 'price_level'], drop_first=False)
    dummy_test = pd.get_dummies(
        test, columns=['country', 'price_level'], drop_first=False)
    # Add the dummy variables to the original dataframe
    train = train.assign(**dummy_train)
    validate = validate.assign(**dummy_validate)
    test = test.assign(**dummy_test)
    # Keep only the columns we need for modeling
    columns_to_keep = ['award', 'sentiment', 'word_count',
                       'price_level_1', 'price_level_2',
                       'price_level_3', 'price_level_4',
                       'country_france', 'country_japan',
                       'country_italy', 'country_usa', 'country_germany']
    train = train[columns_to_keep]
    validate = validate[columns_to_keep]
    test = test[columns_to_keep]

    return train, validate, test


def remove_ngrams(lemmatized: str, ngrams: List[str] = NGRAMS_TO_REMOVE) -> str:
    '''
    removes specified from given lemmatized string 
    ## Parameters
    lemmatized: lemmatized document to be cleaned
    ngrams: list of ngrams to remove from document
    ## Returns
    document with ngrams remove
    '''
    for n in ngrams:
        lemmatized = ''.join(lemmatized.split(n))
    return lemmatized

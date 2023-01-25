import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from IPython.display import Markdown as md
from sklearn.metrics import ConfusionMatrixDisplay

DataType = Union[pd.Series, pd.DataFrame]
ModelType = Union[DecisionTreeClassifier, RandomForestClassifier,
                  LogisticRegression, GradientBoostingClassifier]


def scale(features: DataType, scaler: MinMaxScaler) -> DataType:
    '''
    Fits (if applicable), and scales data with
    # Parameters
    features: either a `Series` or `DataFrame` containing data to be scale.
    scaler: MinMaxScaler used for
    # Returns

    '''
    indexes = features.index
    columns = []
    is_series = False
    if isinstance(features, pd.Series):
        columns.append(features.name)
        features = features.values.reshape(-1, 1)
        is_series = True
    try:
        scaled_data = scaler.transform(features)
    except NotFittedError as e:
        scaler = scaler.fit(features)
        scaled_data = scaler.transform(features)
    if is_series:
        return pd.Series(scaled_data, index=indexes, name='scaled_' + columns[0])
    for c in columns:
        c = 'scaled_' + c
    return pd.DataFrame(scaled_data, index=indexes,
                        columns=columns)


def predict(model: ModelType,
            features: pd.DataFrame,
            target: Union[pd.Series, None] = None,
            result_suffix: str = '') -> pd.Series:
    '''
    Fits (if applicable) and runs predictions on given model.
    # Parameters
    model: a `DecisionTreeClassifier`, `RandomForestClassifier`,
    or `GradientBoostingClassifier` to be modeled on (note)
    # Returns
    `Series` of model predictions
    '''

    y_hat = pd.Series()
    try:
        # gets predictions if model has been fitted
        y_hat = pd.Series(model.predict(features))
    # if model is not fitted, fits model and returns predictions
    except NotFittedError:
        if target is None:
            raise NotFittedError('Model not fit and target not provided')
        model.fit(features, target)
        y_hat = pd.Series(model.predict(features))
    y_hat.name = ('yhat_' + result_suffix if len(result_suffix) > 0 else '')
    # changes indexes to match that of target
    y_hat.index = features.index
    return y_hat

def tf_idf(documents: pd.Series, tfidf: TfidfVectorizer) -> pd.DataFrame:
    # TODO Docstring
    tfidf_docs = np.empty((0, 5))
    try:
        tfidf_docs = tfidf.transform(documents.values)
    except NotFittedError:
        tfidf_docs = tfidf.fit_transform(documents.values)
    return pd.DataFrame(tfidf_docs.todense(), index=documents.index, columns=tfidf.get_feature_names_out())


def get_features_and_target(df: pd.DataFrame,
                            tfidf: TfidfVectorizer) -> Tuple[pd.DataFrame, pd.Series]:

    '''
    scales relevant variables, performs TFIDF, and divides into feature and target
    ## Parameters
    df: `DataFrame` of prepped data
    scaler: `MinMaxScaler` used for scaling
    tfidf: `TfidfVectorizer` used to perform TFIDF
    ## Returns
    Tuple containing the features and the Target
    '''
# TODO Add and scale additional features
    X = tf_idf(df.lemmatized, tfidf)
    y = df.award
    return X, y


def run_train_and_validate(train: pd.DataFrame, validate: pd.DataFrame) -> pd.DataFrame:
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    trainx, trainy = get_features_and_target(train, tfidf=tfidf)
    validx, validy = get_features_and_target(validate, tfidf=tfidf)
    models = [DecisionTreeClassifier(), RandomForestClassifier(),
              LogisticRegression(), GradientBoostingClassifier()]
    ret_df = pd.DataFrame()
    for model in models:
        model_results = {}
        model_name = str(model)
        model_name = model_name[:len(model_name)-2]
        print('Running ' + model_name + ' On Train')
        yhat = predict(model, trainx, trainy)
        model_results['Train'] = accuracy_score(trainy, yhat)
        print('Running ' + model_name + ' On Validate')
        yhat = predict(model, validx)
        model_results['Validate'] = accuracy_score(validy, yhat)
        ret_df[model_name] = model_results
    return ret_df

from typing import Tuple, Union, Dict, List

import numpy as np
import pandas as pd
import pickle
from os.path import isfile
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from IPython.display import Markdown as md

DataType = Union[pd.Series, pd.DataFrame]
ModelType = Union[DecisionTreeClassifier, RandomForestClassifier,
                  LogisticRegression, GradientBoostingClassifier]
NumberType = Union[float, int, np.number]

N_COUNTRIES = 10


def scale(features: DataType, scaler: MinMaxScaler) -> DataType:
    '''
    Fits (if applicable), and scales data with
    # Parameters
    features: either a `Series` or `DataFrame` containing data to be scale.
    scaler: MinMaxScaler used for scaling
    # Returns
    scaled `Series` or `DataFrame`
    '''
    indexes = features.index
    columns = []
    is_series = False
    if isinstance(features, pd.Series):
        columns.append(features.name)
        features = features.values.reshape(-1, 1)
        is_series = True
    else:
        columns = features.columns
    try:
        scaled_data = scaler.transform(features)
    except NotFittedError as e:
        scaler = scaler.fit(features)
        scaled_data = scaler.transform(features)
    if is_series:
        return pd.Series(scaled_data,
                         index=indexes,
                         name='scaled_' + columns[0])
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
    return pd.DataFrame(tfidf_docs.todense(),
                        index=documents.index,
                        columns=tfidf.get_feature_names_out())


def get_features_and_target(df: pd.DataFrame,
                            scaler: MinMaxScaler,
                            tfidf: TfidfVectorizer) -> Tuple[pd.DataFrame,
                                                             pd.Series]:
    '''
    scales relevant variables, performs TFIDF,
    and divides into feature and target
    ## Parameters
    df: `DataFrame` of prepped data
    scaler: `MinMaxScaler` used for scaling
    tfidf: `TfidfVectorizer` used to perform TFIDF
    ## Returns
    Tuple containing the features and the Target
    '''
    tfi_df = tf_idf(df.lemmatized, tfidf)
    top_n_countries = df.country.value_counts(
    )[:N_COUNTRIES].index.to_list()
    language_mask = (~df.country.isin(top_n_countries))
    df.loc[language_mask, 'country'] = 'Other'
    dummies = pd.get_dummies(
        df[['country', 'price_level']])
    scaled_data = scale(df[['word_count', 'sentiment']], scaler)
    X = pd.concat([tfi_df, dummies, scaled_data], axis=1)
    y = df.award
    return X, y


def get_baseline(train: pd.DataFrame) -> md:
    baseline = train.award.value_counts(normalize=True)
    return md('Baseline Value | Baseline'
              '\n---|---'
              f'\n{baseline.index[0]} | {baseline.values[0] * 100:.2f}')


def run_train_and_validate(train: pd.DataFrame,
                           validate: pd.DataFrame) -> pd.DataFrame:
    tfidf = TfidfVectorizer(ngram_range=(1, 2))
    scaler = MinMaxScaler()
    trainx, trainy = get_features_and_target(train, scaler=scaler, tfidf=tfidf)
    validx, validy = get_features_and_target(
        validate, scaler=scaler, tfidf=tfidf)
    models = [DecisionTreeClassifier(max_depth=2),
              RandomForestClassifier(
                  max_depth=5,
                  n_estimators=50,
                  min_samples_leaf=3,
                  random_state=27),
              LogisticRegression(C=.05,
                                 penalty='l1',
                                 random_state=27,
                                 solver='liblinear',
                                 tol=.0001),
              GradientBoostingClassifier(n_estimators=50,
                                         max_depth=4,
                                         min_samples_leaf=4,
                                         random_state=27)]
    ret_df = pd.DataFrame()

    for model in models:
        model_results = {}
        model_name = str(model)
        model_name = model_name.split('(')[0]
        yhat = predict(model, trainx, trainy)
        model_results['Train'] = accuracy_score(trainy, yhat)
        print('Running ' + model_name + ' On Validate')
        yhat = predict(model, validx)
        model_results['Validate'] = accuracy_score(validy, yhat)
        ret_df[model_name] = pd.Series(
            model_results.values(), index=model_results.keys())
    return ret_df.T


def pickle_model(model: ModelType, filename: str) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def unpickle_model(filename: str) -> ModelType:
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        return model


def tune_model(model: ModelType,
               trainx: pd.DataFrame, trainy: pd.DataFrame,
               parameters: Dict[str, List[NumberType]]) -> ModelType:
    scorer = make_scorer(accuracy_score)

    grid_search = GridSearchCV(
        model, parameters, verbose=2, scoring=scorer, n_jobs=5)
    grid_search.fit(trainx, trainy)
    return grid_search.best_params_

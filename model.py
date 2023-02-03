
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
from IPython.display import Markdown as md
import logging
from typing import Dict, List, Tuple, Union
from datatypes import DataType, ModelType, ParameterType
import numpy as np
import pandas as pd
import numpy as np
from datatypes import DataType, ModelType, NumberType
from typing import Dict, List, Tuple, Union
import logging
from IPython.display import Markdown as md

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from datatypes import DataType, ModelType, ParameterType,ClusterType

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


def cluster(df: pd.DataFrame, cluster: ClusterType) -> pd.Series:
    ret_cluster = pd.Series(dtype='int')
    try:
        ret_cluster = cluster.predict(df)
    except NotFittedError as e:
        ret_cluster = cluster.fit_predict(df)
    return ret_cluster

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
                            tfidf: TfidfVectorizer,
                            clusterer: ClusterType) -> Tuple[pd.DataFrame,
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
    scaled_data = scale(df[['word_count', 'sentiment']], scaler)
    df['cluster'] = cluster(
        df[['sentiment', 'latitude', 'longitude']], clusterer)
    dummies = pd.get_dummies(
        df[['country', 'price_level', 'cluster']])
    X = pd.concat([tfi_df, dummies, scaled_data], axis=1)
    y = df.award
    return X, y


def get_baseline(train: pd.DataFrame) -> pd.DataFrame:
    '''
    Generates a `DataFrame` with the baseline (mode) for train
    ## Parameters
    train: DataFrame containing the michelin training data
    ## Returns
    a DataFrame containing the mode of the data
    '''
    baseline = train.award.value_counts(normalize=True)[0]
    return pd.DataFrame([baseline], index=['Baseline'],
                        columns=['Accuracy Score'])


def run_train_and_validate(train: pd.DataFrame,
                           validate: pd.DataFrame,
                           models: List[ModelType],
                           tfidf: TfidfVectorizer,
                           scaler: MinMaxScaler,
                           cluster: ClusterType) -> pd.DataFrame:
    '''
    Runs models on train and validate
    ## Parameters
    train: the training dataset for michelin data
    validate: the out of sample validation dataset for michelin data
    models: classification models to run on Train and Validate
    tfidf: Vectorizer for TFIDF operations
    scaler: scaler for data
    ## Returns
    `DataFrame` containing the accuracy scores for each model
    on the train and validate datasets
    '''
    logging.info('getting features and target for Train')
    trainx, trainy = get_features_and_target(
        train, scaler=scaler, tfidf=tfidf, clusterer=cluster)
    logging.info('getting features and target for Validate')
    validx, validy = get_features_and_target(
        validate, scaler=scaler, tfidf=tfidf, clusterer=cluster)
    ret_df = pd.DataFrame()

    for model in models:
        model_results = {}
        model_name = str(model)
        model_name = model_name.split('(')[0]
        logging.info('Running {model_name} on Train')
        yhat = predict(model, trainx, trainy)
        model_results['Train'] = accuracy_score(trainy, yhat)
        logging.info(f'Running {model_name} on Validate')
        yhat = predict(model, validx)
        model_results['Validate'] = accuracy_score(validy, yhat)
        ret_df[model_name] = pd.Series(
            model_results.values(), index=model_results.keys())
    return ret_df.T


def tune_model(model: ModelType,
               trainx: pd.DataFrame, trainy: pd.DataFrame,
               parameters: Dict[str,
                                List[ParameterType]]) -> Dict[str,
                                                              ParameterType]:
    '''
        Performs Grid Validation on Michelin data
        ## Parameters
        model: model to perform grid validation on
        trainx: features of the training data
        trainy: target of the training data
        parameters: dictionary of lists for each of the
        hyperparameters to tune in the model
        ## Returns
        a `dict` containing the best performing parameters
        '''
    scorer = make_scorer(accuracy_score)

    grid_search = GridSearchCV(
        model, parameters, verbose=2, scoring=scorer, n_jobs=5)
    grid_search.fit(trainx, trainy)
    return grid_search.best_params_


def run_test(test: pd.DataFrame, model: ModelType,
             tfidf: TfidfVectorizer,
             scaler: MinMaxScaler,
             cluster: ClusterType) -> pd.DataFrame:
    '''
    Runs data on test dataset
    ## Parameters
    test: test data from Michelin dataset
    model: final pretrained model
    tfidf: Vectorizer for TFIDF vectorizer
    cluster
    ## Returns
    
    '''
    testx, testy = get_features_and_target(test, scaler, tfidf)
    yhat = predict(model, testx)
    accuracy = accuracy_score(testy, yhat)
    model_name = str(model)
    model_name = model_name.split('(')[0]
    return pd.DataFrame([accuracy],
                        columns=['Accuracy Score'],
                        index=[model_name])


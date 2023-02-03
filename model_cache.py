import pickle
from datatypes import ModelType
from typing import List

def pickle_model(model: ModelType, filename: str) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def unpickle_model(filename: str) -> ModelType:
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        return model


def cache_models(models: List[ModelType]) -> None:
    for m in models:
        model_name = str(m).split('(')[0]
        pickle_model(m, f'data/cached_models/{model_name}.pkl')

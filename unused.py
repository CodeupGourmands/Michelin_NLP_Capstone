import pickle
from datatypes import ModelType


def pickle_model(model: ModelType, filename: str) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def unpickle_model(filename: str) -> ModelType:
    with open(filename, 'rb') as file:
        model = pickle.load(file)
        return model
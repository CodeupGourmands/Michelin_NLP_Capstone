from typing import Union
import numpy as np

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
DataType = Union[pd.Series, pd.DataFrame]
ModelType = Union[DecisionTreeClassifier, RandomForestClassifier,
                  LogisticRegression, GradientBoostingClassifier]
ParameterType = Union[float, int, np.number, str]

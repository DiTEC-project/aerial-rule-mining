import pandas as pd
import math
from sklearn.preprocessing import OneHotEncoder


def is_valid_value(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def encode_categories(X, features):
    """
    :param X:
    :param features:
    :param encoder:
    :return:
    """
    one_hot_encoder = OneHotEncoder(categories="auto", handle_unknown="ignore")
    X_one_hot = pd.DataFrame(one_hot_encoder.fit_transform(X).toarray(), index=X.index)

    X_one_hot.columns = one_hot_encoder.get_feature_names_out(features)
    return X_one_hot

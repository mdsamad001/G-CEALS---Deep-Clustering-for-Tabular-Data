import openml
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from urllib.error import URLError


def do_nothing(*args, **kwargs):
    return None
# monkey patch so that it does not use pq cache files
openml.datasets.functions._get_dataset_parquet = do_nothing


def get_data(id=31, tries=5):
    '''return OHEncoded df'''
    dataset = 0
    err = 0
    
    while(tries>0):
        try:
            dataset = openml.datasets.get_dataset(id)
            break
        except URLError as e:
            tries =- 1
            err = e
            
    if tries<=0:
        raise err

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    
    df = pd.DataFrame(X, columns=attribute_names)
    cat_mask = np.array(categorical_indicator)

    numeric_features = df.columns[~cat_mask]
    categorical_features = df.columns[cat_mask]

    # numeric_transformer = Pipeline(
    #     steps=[
    #         ("scaler", StandardScaler())
    #     ]
    # )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            # ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder='passthrough'
    )

    X_ohe = preprocessor.fit_transform(df)

    return X_ohe, y


def get_data1(id=31, tries=5):
    '''return df with categories are label encoded'''
    dataset = 0
    err = 0
    
    while(tries>0):
        try:
            dataset = openml.datasets.get_dataset(id)
            break
        except URLError as e:
            tries =- 1
            err = e
            
    if tries<=0:
        raise err

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    cat_mask = np.array(categorical_indicator)

    numeric_features = df.columns[~cat_mask]
    categorical_features = df.columns[cat_mask]

    df_copy = df.copy(deep=0)

    for i in categorical_features:
        le = LabelEncoder()
        df_copy[i] = le.fit_transform(df[i].values)
        # df_copy[i] = df_copy[i].astype('int32')

    return df_copy, y, cat_mask


def get_data2(id=31, tries=5):
    '''return df with categories are label encoded'''
    dataset = 0
    err = 0
    
    while(tries>0):
        try:
            dataset = openml.datasets.get_dataset(id)
            break
        except URLError as e:
            tries =- 1
            err = e
            
    if tries<=0:
        raise err

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    cat_mask = np.array(categorical_indicator)

    numeric_features = df.columns[~cat_mask]
    categorical_features = df.columns[cat_mask]

    df_copy = df.copy(deep=0)

    for i in categorical_features:
        le = LabelEncoder()
        df_copy[i] = le.fit_transform(df[i].values)
        # df_copy[i] = df_copy[i].astype('int32')

    return df_copy, y, cat_mask, dataset



def get_data_raw(id=31, tries=5):
    '''return df with categories are label encoded'''
    dataset = 0
    err = 0
    
    while(tries>0):
        try:
            dataset = openml.datasets.get_dataset(id)
            break
        except URLError as e:
            tries =- 1
            err = e
            
    if tries<=0:
        raise err

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    cat_mask = np.array(categorical_indicator)

    numeric_features = df.columns[~cat_mask]
    categorical_features = df.columns[cat_mask]

    df_copy = df.copy(deep=0)

    return df_copy, y, cat_mask
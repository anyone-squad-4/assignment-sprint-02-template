from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned-up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarray
        val : np.ndarray
        test : np.ndarray
    """

    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    categorical_cols = working_train_df.select_dtypes(include=["object"]).columns
    binary_cols = [col for col in categorical_cols if working_train_df[col].nunique() == 2]
    multi_cols = [col for col in categorical_cols if working_train_df[col].nunique() > 2]


    ordinal_encoder = OrdinalEncoder()
    working_train_df[binary_cols] = ordinal_encoder.fit_transform(working_train_df[binary_cols])
    working_val_df[binary_cols] = ordinal_encoder.transform(working_val_df[binary_cols])
    working_test_df[binary_cols] = ordinal_encoder.transform(working_test_df[binary_cols])
    
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    onehot_encoded_train = onehot_encoder.fit_transform(working_train_df[multi_cols])
    onehot_encoded_val = onehot_encoder.transform(working_val_df[multi_cols])
    onehot_encoded_test = onehot_encoder.transform(working_test_df[multi_cols])
    onehot_cols = onehot_encoder.get_feature_names_out(multi_cols)
    onehot_train_df = pd.DataFrame(onehot_encoded_train, columns=onehot_cols, index=working_train_df.index)
    onehot_val_df = pd.DataFrame(onehot_encoded_val, columns=onehot_cols, index=working_val_df.index)
    onehot_test_df = pd.DataFrame(onehot_encoded_test, columns=onehot_cols, index=working_test_df.index)

    # Drop original multi-category columns and concatenate one-hot encoded columns
    working_train_df = pd.concat([working_train_df.drop(columns=multi_cols), onehot_train_df], axis=1)
    working_val_df = pd.concat([working_val_df.drop(columns=multi_cols), onehot_val_df], axis=1)
    working_test_df = pd.concat([working_test_df.drop(columns=multi_cols), onehot_test_df], axis=1)

    imputer = SimpleImputer(strategy="median")
    working_train_df = pd.DataFrame(imputer.fit_transform(working_train_df), columns=working_train_df.columns, index=working_train_df.index)
    working_val_df = pd.DataFrame(imputer.transform(working_val_df), columns=working_val_df.columns, index=working_val_df.index)
    working_test_df = pd.DataFrame(imputer.transform(working_test_df), columns=working_test_df.columns, index=working_test_df.index)

    scaler = MinMaxScaler()
    working_train_df = pd.DataFrame(scaler.fit_transform(working_train_df), columns=working_train_df.columns, index=working_train_df.index)
    working_val_df = pd.DataFrame(scaler.transform(working_val_df), columns=working_val_df.columns, index=working_val_df.index)
    working_test_df = pd.DataFrame(scaler.transform(working_test_df), columns=working_test_df.columns, index=working_test_df.index)

    train_array = working_train_df.to_numpy()
    val_array = working_val_df.to_numpy()
    test_array = working_test_df.to_numpy()

    print("Processed train data shape: ", train_array.shape)
    print("Processed val data shape: ", val_array.shape)
    print("Processed test data shape: ", test_array.shape, "\n")

    return train_array, val_array, test_array

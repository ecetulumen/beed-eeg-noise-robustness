"""Dataset loading, validation, splitting, scaling, and clipping."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

from config import (
    CLIP_VALUE,
    RANDOM_STATE,
    SCALER_TYPE,
    TEST_SIZE,
    USE_FEATURE_CLIPPING,
)
from utils import print_section


def apply_clipping_if_needed(values):
    if USE_FEATURE_CLIPPING:
        return np.clip(values, -CLIP_VALUE, CLIP_VALUE)
    return values


def create_scaler():
    if SCALER_TYPE.lower() == "robust":
        print("Scaler: RobustScaler")
        return RobustScaler()
    print("Scaler: StandardScaler")
    return StandardScaler()


def load_dataset(data_path):
    print_section("DATASET INFORMATION")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Set BEED_DATA_PATH to use another file."
        )

    dataframe = pd.read_csv(data_path)
    if "y" not in dataframe.columns:
        raise ValueError("Target column must be named 'y'.")
    if dataframe.isna().any().any():
        raise ValueError("The dataset contains missing values.")

    dataframe["y"] = dataframe["y"].astype(int)
    features = dataframe.drop(columns="y")

    non_numeric = features.select_dtypes(exclude=np.number).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(dataframe["y"])

    print("Dataset shape:", dataframe.shape)
    print("Features:", features.shape[1])
    print("Classes:", encoder.classes_.tolist())
    print("Class distribution:\n", dataframe["y"].value_counts().sort_index())

    return features, labels, encoder


def split_and_scale(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    scaler = create_scaler()
    X_train_scaled = apply_clipping_if_needed(scaler.fit_transform(X_train))
    X_test_scaled = apply_clipping_if_needed(scaler.transform(X_test))

    print("Train shape:", X_train_scaled.shape)
    print("Test shape:", X_test_scaled.shape)
    print("Scaling completed without data leakage.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


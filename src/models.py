"""Model definitions and training routines."""

import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from xgboost import XGBClassifier

from config import (
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_VALIDATION_SPLIT,
    RANDOM_STATE,
)
from utils import print_section


def create_ml_models(num_classes):
    models = {
        "SVM": SVC(
            kernel="rbf",
            C=80.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            class_weight="balanced_subsample",
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    xgb_common = dict(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=1,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.5,
        reg_alpha=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if num_classes == 2:
        models["XGBoost"] = XGBClassifier(
            **xgb_common,
            objective="binary:logistic",
            eval_metric="logloss",
        )
    else:
        models["XGBoost"] = XGBClassifier(
            **xgb_common,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=num_classes,
        )

    return models


def build_mlp(input_dim, num_classes):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.35),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.30),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.20),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ml_models(X_train, y_train, training_type, num_classes):
    print_section(f"{training_type} - MACHINE LEARNING TRAINING")
    trained_models = {}

    for model_name, model in create_ml_models(num_classes).items():
        print(f"{model_name} training started...")
        model.fit(X_train, y_train)
        trained_models[model_name] = {
            "model": model,
            "type": "ml",
            "training_type": training_type,
        }
        print(f"{model_name} training completed.")

    return trained_models


def train_mlp(X_train, y_train, input_dim, num_classes, training_type):
    print_section(f"{training_type} - MLP TRAINING")
    model = build_mlp(input_dim, num_classes)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5
        ),
    ]
    history = model.fit(
        X_train,
        to_categorical(y_train, num_classes=num_classes),
        epochs=MLP_EPOCHS,
        batch_size=MLP_BATCH_SIZE,
        validation_split=MLP_VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )
    model_item = {
        "model": model,
        "type": "mlp",
        "training_type": training_type,
    }
    return model_item, history


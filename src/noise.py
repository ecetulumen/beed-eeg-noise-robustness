"""AWGN generation and noise-augmented training data creation."""

import numpy as np

from config import RANDOM_STATE, ROBUST_REPEAT_MAP, ROBUST_TRAIN_SNR_LEVELS
from data_processing import apply_clipping_if_needed


def add_awgn_noise(values, snr_db, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    noisy_values = values.copy()

    signal_power = np.mean(noisy_values ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0,
        scale=np.sqrt(noise_power),
        size=noisy_values.shape,
    )
    return apply_clipping_if_needed(noisy_values + noise)


def create_robust_training_data(X_train, y_train):
    feature_parts = [X_train]
    label_parts = [y_train]

    for snr in ROBUST_TRAIN_SNR_LEVELS:
        for repeat in range(ROBUST_REPEAT_MAP.get(snr, 1)):
            seed = RANDOM_STATE + snr * 100 + repeat
            feature_parts.append(add_awgn_noise(X_train, snr, seed))
            label_parts.append(y_train)

    return np.vstack(feature_parts), np.concatenate(label_parts)


"""Shared display, saving, and reproducibility helpers."""

import random

import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def print_section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def save_df(dataframe, filename):
    path = OUTPUT_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_fig(filename):
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


def display_and_save(dataframe, title, filename):
    print_section(title)
    display(dataframe)
    save_df(dataframe, filename)


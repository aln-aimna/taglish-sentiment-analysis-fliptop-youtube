"""Loading the labelled data and making the fixed train / test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .cleaning import basic_clean, for_vader, normalize


def add_text_columns(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Add text_clean / text_vader / text_norm / tokens for any dataframe with raw text
    (used for the hand-written hard cases, which don't come from comments_clean.csv)."""
    d = df.copy()
    d["text_clean"] = d[text_col].map(basic_clean)
    d["text_vader"] = d[text_col].map(for_vader)
    d["text_norm"] = d[text_col].map(normalize)
    return d


def make_split(seed: int = config.SEED, test_size: int = config.TEST_SIZE, save: bool = True):
    """Stratified split of labeled_comments.csv into train and a fixed 300-row test set."""
    lab = pd.read_csv(config.LABELED_FILE)
    train, test = train_test_split(
        lab, test_size=test_size, stratify=lab["label"], random_state=seed
    )
    train = train.sort_values("label_id").reset_index(drop=True)
    test = test.sort_values("label_id").reset_index(drop=True)
    if save:
        train.to_csv(config.TRAIN_FILE, index=False)
        test.to_csv(config.TEST_FILE, index=False)
    return train, test


def load_split():
    """Read the saved split (every notebook after 01 uses this, so all models see the same rows)."""
    if not config.TEST_FILE.exists():
        make_split()
    train = pd.read_csv(config.TRAIN_FILE, keep_default_na=False)
    test = pd.read_csv(config.TEST_FILE, keep_default_na=False)
    return train, test


def load_hard_cases() -> pd.DataFrame:
    return add_text_columns(pd.read_csv(config.HARD_FILE))

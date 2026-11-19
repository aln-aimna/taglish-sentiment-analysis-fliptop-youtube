"""Scoring helpers shared by every tier, so all models are measured the same way."""
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from . import config

LABELS = config.LABELS
METRIC_COLS = ["model", "tier", "accuracy", "macro_f1", "hard_acc", "sec_per_1k", "cost_per_1k_usd"]


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


def report(y_true, y_pred, title: str = ""):
    """Print per-class precision / recall / F1."""
    if title:
        print(f"== {title} ==")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))


def timed_predict(predict_fn, texts) -> tuple:
    """Run predict_fn over a list of texts. Returns (predictions, seconds per 1,000 texts)."""
    texts = list(texts)
    t0 = time.perf_counter()
    preds = list(predict_fn(texts))
    secs = time.perf_counter() - t0
    return preds, secs / len(texts) * 1000


def plot_confusion(y_true, y_pred, title: str, ax=None, save: bool = True):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    if save:
        fname = "cm_" + "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")
        ax.figure.tight_layout()
        ax.figure.savefig(config.CHARTS / f"{fname}.png", dpi=150)
    return ax


def _upsert_column(path, key_col, keys, col, values):
    if path.exists():
        df = pd.read_csv(path, keep_default_na=False)
    else:
        df = pd.DataFrame({key_col: list(keys)})
    if len(df) != len(keys) or list(df[key_col]) != list(keys):
        df = pd.DataFrame({key_col: list(keys)})           # split changed -> start over
    df[col] = list(values)
    df.to_csv(path, index=False)


def save_run(model: str, tier: str, test_df, test_pred, hard_df, hard_pred,
             sec_per_1k: float, cost_per_1k_usd: float = 0.0) -> dict:
    """Store this model's predictions and one row of metrics (re-running a model overwrites it)."""
    _upsert_column(config.PREDICTIONS_FILE, "label_id", test_df["label_id"], model, test_pred)
    _upsert_column(config.HARD_PREDICTIONS_FILE, "hard_id", hard_df["hard_id"], model, hard_pred)

    row = {"model": model, "tier": tier, **scores(test_df["label"], test_pred),
           "hard_acc": accuracy_score(hard_df["label"], hard_pred),
           "sec_per_1k": sec_per_1k, "cost_per_1k_usd": cost_per_1k_usd}
    if config.METRICS_FILE.exists():
        m = pd.read_csv(config.METRICS_FILE)
        m = m[m["model"] != model]
        m = pd.concat([m, pd.DataFrame([row])], ignore_index=True)
    else:
        m = pd.DataFrame([row])
    m[METRIC_COLS].to_csv(config.METRICS_FILE, index=False)
    return row


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(config.METRICS_FILE)

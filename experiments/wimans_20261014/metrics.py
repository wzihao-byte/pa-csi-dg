"""Explicit fixed-nine-class metrics; reduce each run before aggregating runs."""
from __future__ import annotations

import numpy as np

from .adapter import CLASSES


def confusion_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Expected equally sized nonempty one-dimensional predictions")
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Labels and predictions must be integer class IDs")
    if min(labels.min(), predictions.min()) < 0 or max(labels.max(), predictions.max()) > 8:
        raise ValueError("Class IDs must be 0..8")
    confusion = np.bincount(9 * labels + predictions, minlength=81).reshape(9, 9)
    support, predicted, tp = confusion.sum(1), confusion.sum(0), np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros(9, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(9, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(9), where=precision + recall > 0)
    return {"accuracy": float(tp.sum() / len(labels)), "f1_macro": float(f1.mean()),
            "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()),
            "confusion_matrix": confusion.tolist(),
            "per_class": [{"class_id": c, "activity": CLASSES[c], "support": int(support[c]),
                           "precision": float(precision[c]), "recall": float(recall[c]), "f1": float(f1[c])}
                          for c in range(9)]}

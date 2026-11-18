"""Score headlines with FinBERT (ProsusAI/finbert, Apache-2.0).

The original FinBERT repo uses the old ``pytorch_pretrained_bert`` library.
This module loads the same published weights through Hugging Face
``transformers`` instead, which is the migration the original authors list
as a to-do.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import pandas as pd

MODEL_NAME = "ProsusAI/finbert"
LABELS = ("positive", "negative", "neutral")


class Scorer(Protocol):
    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, 3) array of probabilities in LABELS order."""


class FinBertScorer:
    """Batched FinBERT inference. Uses a GPU if one is available."""

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        # Reorder the model's own label ids into LABELS order.
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.order = [next(i for i, l in id2label.items() if l == name) for name in LABELS]

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1)
            out.append(probs[:, self.order].cpu().numpy())
        return np.vstack(out) if out else np.empty((0, 3))


def score_headlines(df: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    """Add p_positive, p_negative, p_neutral, label and a single score.

    ``score = p_positive - p_negative`` runs from -1 (clearly bad news) to
    +1 (clearly good news); neutral headlines land near 0.
    """
    out = df.copy()
    unique = out["title"].drop_duplicates().tolist()
    probs = scorer(unique) if unique else np.empty((0, 3))
    table = pd.DataFrame(probs, columns=[f"p_{l}" for l in LABELS])
    table["title"] = unique
    out = out.merge(table, on="title", how="left")
    out["label"] = out[[f"p_{l}" for l in LABELS]].idxmax(axis=1).str.removeprefix("p_")
    out["score"] = out["p_positive"] - out["p_negative"]
    return out

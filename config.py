"""Shared configuration for the fraud detection project."""

from pathlib import Path

DATA_PATH = Path("creditcard.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Business framing: a missed fraud case (false negative) is assumed to cost
# roughly 10x more than a false alarm (false positive, a blocked legit
# transaction). This ratio drives threshold tuning in evaluate.py.
COST_FALSE_NEGATIVE = 10
COST_FALSE_POSITIVE = 1

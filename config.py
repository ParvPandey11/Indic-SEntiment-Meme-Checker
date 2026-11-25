"""Central settings. Edit paths here if your folders differ."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")            # train.csv / val.csv / test.csv (searched recursively)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RUNS_DIR = os.path.join(ROOT, "runs")

IMG_DIRS = {
    "train": os.path.join(DATA_DIR, "trainImages"),
    "val": os.path.join(DATA_DIR, "valImages"),
    "test": os.path.join(DATA_DIR, "testImages"),
}

SEED = 42
LABELS = ["negative", "neutral", "positive"]       # index = class id
# Only used if the CSV stores sentiment as numbers. CHECK against the dataset documentation.
NUMERIC_MAP = {-2: 0, -1: 0, 0: 1, 1: 2, 2: 2}

import os
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent
ROOT = _CODE_DIR.parent if _CODE_DIR.name == "src" else _CODE_DIR
CSV_PATH = ROOT / "pets.csv"
PETS256_ZIP = ROOT / "pets256.zip"
PETS_ORIGINAL_ZIP = ROOT / "pets_original.zip"
DATA_DIR = ROOT / "data"
PETS256_DIR = DATA_DIR / "pets256"
PETS_ORIGINAL_DIR = DATA_DIR / "pets_original"
OUTPUT_DIR = ROOT / "outputs"
FEATURES_DIR = OUTPUT_DIR / "features"
SPLITS_DIR = OUTPUT_DIR / "splits"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
RETRIEVAL_EXAMPLES_DIR = OUTPUT_DIR / "retrieval_examples"
MPLCONFIGDIR = ROOT / ".cache" / "matplotlib"
XDG_CACHE_HOME = ROOT / ".cache"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

RANDOM_SEED = 42

DESCRIPTORS = ["gch", "lbp", "glcm", "hog", "correlogram", "gabor"]
COMBINATIONS = {
    "gch": ["gch"],
    "lbp": ["lbp"],
    "glcm": ["glcm"],
    "hog": ["hog"],
    "correlogram": ["correlogram"],
    "gabor": ["gabor"],
    "gch+lbp": ["gch", "lbp"],
    "gch+glcm": ["gch", "glcm"],
    "gch+lbp+glcm": ["gch", "lbp", "glcm"],
    "gch+lbp+glcm+gabor": ["gch", "lbp", "glcm", "gabor"],
    "all": ["gch", "lbp", "glcm", "hog", "correlogram", "gabor"],
}

CLASSIFICATION_GRID = [
    {"kernel": "linear", "C": 0.1},
    {"kernel": "linear", "C": 1.0},
    {"kernel": "linear", "C": 10.0},
    {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
    {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    {"kernel": "rbf", "C": 0.1, "gamma": 0.01},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.01},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.01},
    {"kernel": "rbf", "C": 0.1, "gamma": 0.001},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.001},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.001},
]

BOVW_K = 200
BOVW_FAST_K = 20
BOVW_PATCH_SIZE = 16
BOVW_PATCH_STEP = 16
BOVW_MAX_PATCHES = 50_000


def ensure_output_dirs():
    for path in [
        DATA_DIR,
        OUTPUT_DIR,
        FEATURES_DIR,
        SPLITS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        RETRIEVAL_EXAMPLES_DIR,
        MPLCONFIGDIR,
        XDG_CACHE_HOME / "fontconfig",
    ]:
        path.mkdir(parents=True, exist_ok=True)

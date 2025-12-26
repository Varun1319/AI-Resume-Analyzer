# config.py

import os
from pathlib import Path

# Project root (directory containing app.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# ----------------- Models & Paths ----------------- #
MODELS_DIR = PROJECT_ROOT / "models"

# Meta-ranker model path (GradientBoosting / XGBoost, etc.)
RANKER_MODEL_NAME = "ranker_gb.pkl"  # or "ranker_xgb.pkl" if you stuck with that name
RANKER_MODEL_PATH = MODELS_DIR / RANKER_MODEL_NAME

# BERT / SentenceTransformer model name
BERT_MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------- Scoring defaults ----------------- #
DEFAULT_SHORTLIST_THRESHOLD = 0.70  # 70%

# ----------------- OpenAI / API keys ----------------- #
# We don't hardcode the key; we just define the env var name.
OPENAI_API_ENV_VAR = "OPENAI_API_KEY"
# config.py

import os
from pathlib import Path

# Project root (directory containing app.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# ----------------- Models & Paths ----------------- #
MODELS_DIR = PROJECT_ROOT / "models"

# Meta-ranker model path (GradientBoosting / XGBoost, etc.)
RANKER_MODEL_NAME = "ranker_gb.pkl"  # or "ranker_xgb.pkl" if you stuck with that name
RANKER_MODEL_PATH = MODELS_DIR / RANKER_MODEL_NAME

# BERT / SentenceTransformer model name
BERT_MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------- Scoring defaults ----------------- #
DEFAULT_SHORTLIST_THRESHOLD = 0.70  # 70%

# ----------------- OpenAI / API keys ----------------- #
# We don't hardcode the key; we just define the env var name.
OPENAI_API_ENV_VAR = "OPENAI_API_KEY"

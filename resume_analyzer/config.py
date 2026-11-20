# config.py

from pathlib import Path

# ----------------- Paths ----------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

SKILLS_PATH = DATA_DIR / "skills.txt"
RAW_DATASET_PATH = DATA_DIR / "resume_data.csv"  # the Kaggle resume dataset
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
RANKER_MODEL_PATH = MODELS_DIR / "ranker_gb.pkl"

# ----------------- Models ----------------- #
# Sentence-BERT model for semantic similarity
BERT_MODEL_NAME = "all-MiniLM-L6-v2"

# OpenAI models for resume feedback (the API key is entered in the app)
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

# ----------------- Resume chunking & retrieval ----------------- #
CHUNK_SIZE = 700
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 3

# ----------------- Scoring ----------------- #
STRONG_MATCH_THRESHOLD = 0.70
MODERATE_MATCH_THRESHOLD = 0.40

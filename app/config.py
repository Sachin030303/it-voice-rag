
import os

DATA_DIR = os.getenv("DATA_DIR", "./data")

# Gemini embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

# Keep the generation model names already verified for your API account.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "gemini-3.5-flash-lite")

TOP_K = int(os.getenv("TOP_K", "5"))
SIM_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))

CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "900"))
CHUNK_OVERLAP_PARAS = 1
MAX_HISTORY_TURNS = 6

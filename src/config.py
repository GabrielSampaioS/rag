from pathlib import Path

# ==============================
# CONFIGURAÇÕES
# ==============================

BASE_DIR = (Path(__file__).resolve().parent/"docs")
PDF_PATH = BASE_DIR / "CoverLetter.pdf"
CHROMA_DIR = BASE_DIR / "chroma_cover_letter"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3.5:4b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
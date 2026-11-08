from pathlib import Path

# ==============================
# CONFIGURAÇÕES
# ==============================

# Raiz do projeto (diretório acima de src)
BASE_DIR = Path(__file__).resolve().parent.parent

# Documentos
PDF_PATH = BASE_DIR / "data" / "documents" / "CoverLetter.pdf"

# Banco vetorial
CHROMA_DIR = BASE_DIR / "data" / "vector_store" / "chroma"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen3.5:4b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
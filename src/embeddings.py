from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

# ==============================
# DIVISÃO EM CHUNKS
# ==============================

def split_documents(documents):
    """Divide os documentos em chunks."""

    print("\nIniciando divisão dos documentos em chunks...\n")

    ignored_metadata = [
        "creator",
        "producer",
        "creationdate",
        "page_label",
        "total_pages"
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print(f"[INFO] Documentos recebidos: {len(documents)}")
    print(f"[INFO] Chunks gerados: {len(chunks)}")

    return chunks


def show_chunk_info(documents, chunks):
    """Exibe informações sobre os chunks gerados."""

    print("\n========== INFORMAÇÕES ==========")
    print(f"Total de páginas: {len(documents)}")
    print(f"Total de chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Caracteres: {len(chunk.page_content)}")
        print(f"Página: {chunk.metadata.get('page', 'N/A')}")
        print(f"Conteúdo:\n{chunk.page_content}")


# ==============================
# EMBEDDINGS
# ==============================

def create_embeddings():

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


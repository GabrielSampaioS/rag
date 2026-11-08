from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

# ==============================
# DIVISÃO EM CHUNKS
# ==============================

def split_documents(documents):
    """Divide os documentos em chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


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


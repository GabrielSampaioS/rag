from langchain_chroma import Chroma
from src.config import CHROMA_DIR
from tqdm import tqdm

# ==============================
# BANCO VETORIAL
# ==============================


def create_vector_store(embeddings):

    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )


def store_chunks(vectorstore, chunks):
    """Armazena os chunks caso o banco esteja vazio."""

    count = vectorstore._collection.count()

    if count == 0:

        print("\nBanco vazio. Gerando embeddings...\n")

        batch_size = 10

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Gerando embeddings",
            unit="lote"
        ):

            batch = chunks[i:i + batch_size]

            vectorstore.add_documents(batch)

        print("\nEmbeddings armazenados com sucesso!")

    else:

        print(f"Banco existente: {count} documentos.")


# ==============================
# RETRIEVER 
# ==============================

def create_retriever(vectorstore):
    """Cria mecanismo de recuperação por similaridade."""
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
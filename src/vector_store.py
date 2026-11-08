from langchain_chroma import Chroma
from src.config import CHROMA_DIR
from tqdm import tqdm
from src.metadata_enricher import generate_metadata
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

def enrich_chunks(chunks):
    """Enriquece os chunks com metadados semânticos."""

    print("\nIniciando geração de metadados...\n")

    ignored_metadata = [
        "creator",
        "producer",
        "creationdate",
        "page_label",
        "total_pages"
    ]

    for i, chunk in enumerate(
        tqdm(chunks, desc="Gerando metadados", unit="chunk"),
        start=1
    ):
        # Remove metadados técnicos desnecessários
        for key in ignored_metadata:
            chunk.metadata.pop(key, None)
        # Gera metadados semânticos
        semantic_metadata = generate_metadata(
            chunk.page_content
        )

        chunk.metadata.update(semantic_metadata)


    print("\n[INFO] Metadados gerados com sucesso!")
    print(f"[INFO] Total de chunks enriquecidos: {len(chunks)}")

    return chunks


# ==============================
# RETRIEVER 
# ==============================

def create_retriever(vectorstore):
    """Cria mecanismo de recuperação por similaridade."""
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
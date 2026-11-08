from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from tqdm import tqdm

from src.config import CHROMA_DIR
from src.metadata_classifier import (
    enrich_chunks as apply_semantic_metadata,
    retrieve_with_filter,
)

from src.helper.print_section import print_info, print_subsection

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

        print_info("Banco vazio. Gerando embeddings")

        batch_size = 10

        for i in tqdm(
            range(0, len(chunks), batch_size),
            desc="Gerando embeddings",
            unit="lote"
        ):

            batch = chunks[i:i + batch_size]

            vectorstore.add_documents(batch)

        print_info("Embeddings armazenados com sucesso!")

    else:

        print_info(f"Banco existente: {count} documentos.")


def enrich_chunks(chunks):
    """Enriquece os chunks com metadados semânticos."""

    print_info("Iniciando geração de metadados...")

    ignored_metadata = [
        "creator",
        "producer",
        "creationdate",
        "page_label",
        "total_pages"
    ]

    # Remove metadados técnicos desnecessários
    for chunk in chunks:
        for key in ignored_metadata:
            chunk.metadata.pop(key, None)

    # Classifica todos os chunks de uma vez, na ordem do splitter
    chunks = apply_semantic_metadata(chunks)

    print_info("Metadados gerados com sucesso!")
    print_info(f"Total de chunks enriquecidos: {len(chunks)}")

    return chunks


# ==============================
# RETRIEVER
# ==============================


class FilteredRetriever(BaseRetriever):
    """Busca nas categorias prováveis da pergunta e completa com busca normal."""

    vectorstore: Any
    k: int = 6
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        return retrieve_with_filter(self.vectorstore, query, k=self.k)


def create_retriever(vectorstore, k=6):
    """Cria mecanismo de recuperação com filtro por categoria."""

    return FilteredRetriever(vectorstore=vectorstore, k=k)
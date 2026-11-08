from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import re

from src.helper import print_section
from src.helper.print_section import print_info

# ==============================
# CARREGAMENTO DO DOCUMENTO
# ==============================

def load_pdf(pdf_path: Path):
    """Carrega o conteúdo do PDF."""

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        raise ValueError(f"Nenhum conteúdo encontrado: {pdf_path}")

    return documents

def load_documents():
    """Carrega e prepara os documentos da pasta docs."""
    docs_dir = Path(__file__).resolve().parent.parent / "data" / "documents"
    print_info(f"Procurando PDFs em: {docs_dir}")
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("Nenhum PDF encontrado na pasta docs.")
    documents = []

    for pdf_path in pdf_files:
        loaded = load_pdf(pdf_path)
        documents.extend(loaded)

    documents = clean_documents(documents)

    print_info(f"Total de páginas carregadas: {len(documents)}")

    return documents


def clean_documents(documents):
    """Remove quebras de linha e espaços excessivos do PDF."""

    for doc in documents:

        # Substitui sequências de espaços e quebras de linha por um espaço.
        doc.page_content = re.sub(
            r"\s+",
            " ",
            doc.page_content
        ).strip()

    return documents

def inspect_documents(documents):

    for i, doc in enumerate(documents, start=1):

        print_section(f"\n========== PÁGINA {i} ==========")

        print_info(repr(doc.page_content[:1000]))
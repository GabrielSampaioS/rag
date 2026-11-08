from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from src.helper.print_section import print_info, print_subsection

# ==============================
# DIVISÃO EM CHUNKS
# ==============================

def split_documents(documents):
    """Divide os documentos em chunks."""

    print_info("Iniciando divisão dos documentos em chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print_info(f"Documentos recebidos: {len(documents)}")
    print_info(f"Chunks gerados: {len(chunks)}")

    return chunks


def show_chunk_info(documents, chunks):
    """Exibe informações sobre os chunks gerados."""

    print_subsection("INFORMAÇÕE")
    print_subsection(f"Total de páginas: {len(documents)}")
    print_subsection(f"Total de chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print_subsection(f"\nChunk {i}")
        print_subsection(f"Caracteres: {len(chunk.page_content)}")
        print_subsection(f"Página: {chunk.metadata.get('page', 'N/A')}")
        print_subsection(f"Conteúdo:\n{chunk.page_content}")


# ==============================
# EMBEDDINGS
# ==============================

def create_embeddings():

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


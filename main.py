
from pathlib import Path
import re
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


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


# ==============================
# BANCO VETORIAL
# ==============================

def create_vectorstore(embeddings):

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

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


# ==============================
# MODELO DE LINGUAGEM
# ==============================

def create_llm():

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )


# ==============================
# CONSTRUÇÃO DA CADEIA RAG
# ==============================

def build_qa_chain(retriever):

    print("\n[INFO] Entrou em build_qa_chain()")
    llm = create_llm()
    print("[INFO] LLM criado com sucesso. Montando RetrievalQA...")


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print("[SUCESSO] RetrievalQA criado com sucesso!")
    return qa_chain


# ==============================
# EXECUÇÃO DE PERGUNTAS
# ==============================

def ask_and_print(qa_chain: RetrievalQA, pergunta: str):
     
    print("\n[INFO] Entrou em ask_and_print()")
    print("[INFO] Enviando pergunta ao agente...")


    resposta = qa_chain.invoke({"query": pergunta})

    print("\nPergunta:")
    print(pergunta)

    print("\nResposta do Agente:")
    print(resposta["result"])

    print("\nTrechos utilizados como contexto:\n")

    for i, doc in enumerate(resposta["source_documents"], start=1):

        print(f"--- Trecho {i} ---")
        print(f"Fonte: {doc.metadata.get('source', 'Desconhecido')}")
        print(f"Página: {doc.metadata.get('page', 'N/A')}")
        print(doc.page_content)

def inspect_documents(documents):

    for i, doc in enumerate(documents, start=1):

        print(f"\n========== PÁGINA {i} ==========")

        print(repr(doc.page_content[:1000]))
        
# ==============================
# PIPELINE PRINCIPAL
# ==============================

def main():

    # 1. Carregar PDF
    documents = load_pdf(PDF_PATH)
    clean_documents(documents)
    #inspect_documents(documents)

    # 2. Gerar chunks
    chunks = split_documents(documents)

    # 3. Exibir informações
    #show_chunk_info(documents, chunks)

    # ==================================
    # MODO DE TESTE: SOMENTE CHUNKS
    # ==================================

  

    # ==================================
    # PIPELINE RAG COMPLETO
    # ==================================

    embeddings = create_embeddings()

    vectorstore = create_vectorstore(embeddings)

    store_chunks(vectorstore, chunks)

    retriever = create_retriever(vectorstore)

    qa_chain = build_qa_chain(retriever)

    ask_and_print(qa_chain, "Qual é o nome do candidato?")
    ask_and_print(qa_chain, "Qual é o diferencial dele?")
    return


if __name__ == "__main__":
    main()

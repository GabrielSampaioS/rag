from src.document_loader import load_documents
from src.embeddings import create_embeddings, split_documents, show_chunk_info
from src.vector_store import create_vector_store, create_retriever, store_chunks
from src.qa_chain import build_qa_chain, ask_and_print


def main():

    print("\n========== INICIANDO RAG ==========\n")

    # 1. Carregar documentos
    print("[1/4] Carregando documentos...")
    documents = load_documents()
    chunks = split_documents(documents)
    #show_chunk_info(documents, chunks)
  

    # 2. Criar embeddings
    print("[2/4] Preparando embeddings...")
    embeddings = create_embeddings()

    # 3. Criar banco vetorial e retriever
    print("[3/4] Preparando banco vetorial...")
    vectorstore = create_vector_store(embeddings)
    store_chunks(vectorstore, chunks)
    retriever = create_retriever(vectorstore)

    # 4. Construir cadeia de perguntas
    print("[4/4] Construindo QA Chain...")
    qa_chain = build_qa_chain(retriever)

    print("\n========== RAG PRONTO ==========\n")

    # Interação com usuário
    while True:

        pergunta = input("\nDigite sua pergunta (ou 'q' para sair): ")

        if pergunta.lower() == "q":
            print("Encerrando aplicação...")
            break

        ask_and_print(qa_chain, pergunta)
    return

if __name__ == "__main__":
    main()
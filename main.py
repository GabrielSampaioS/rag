from src.document_loader import load_documents
from src.embeddings import create_embeddings, split_documents, show_chunk_info
from src.helper.print_section import print_section, print_subsection
from src.vector_store import create_vector_store, create_retriever, store_chunks, enrich_chunks
from src.qa_chain import build_qa_chain, ask_and_print


def main():

    print_section("INICIANDO RAG")

    # 1. Carregar documentos
    print_subsection("[1/4] Carregando documentos...")
    documents = load_documents()
    chunks = split_documents(documents)
    chunks = enrich_chunks(chunks)
    # show_chunk_info(documents, chunks)

    # 2. Criar embeddings
    print_subsection("[2/4] Preparando embeddings...")
    embeddings = create_embeddings()

    # 3. Criar banco vetorial e retriever
    print_subsection("[3/4] Preparando banco vetorial...")
    vectorstore = create_vector_store(embeddings)
    store_chunks(vectorstore, chunks)  # só insere se o banco estiver vazio
    retriever = create_retriever(vectorstore, k=8)

    # 4. Construir cadeia de perguntas
    print_subsection("[4/4] Construindo QA Chain...")
    qa_chain = build_qa_chain(retriever)

    print_section("RAG PRONTO")

    # Interação com usuário
    while True:

        pergunta = input("\nDigite sua pergunta (ou 'q' para sair): ")

        if pergunta.lower() == "q":
            print("Encerrando aplicação...")
            break

        ask_and_print(qa_chain, pergunta)


if __name__ == "__main__":
    main()
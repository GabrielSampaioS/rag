from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from src.config import LLM_MODEL


# ==============================
# MODELO DE LINGUAGEM
# ==============================

def create_llm():

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
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


def ask_and_print(qa_chain: RetrievalQA, pergunta: str):
    """Executa uma pergunta e apresenta a resposta."""

    print("\n[INFO] Executando consulta...")

    resposta = qa_chain.invoke({"query": pergunta})

    print("\n========== RESPOSTA ==========")
    print(resposta["result"])

    print("\n========== FONTES ==========")

    for doc in resposta["source_documents"]:
        arquivo = doc.metadata.get("source", "Arquivo desconhecido")
        pagina = doc.metadata.get("page", -1) + 1
        print(f"\nArquivo: {arquivo}")
        print(f"Página: {pagina}")
        print(f"Trecho: {doc.page_content[:300]}...")

from pathlib import Path
import chromadb

from src.helper.print_section import print_info, print_subsection

# Raiz do projeto (diretório acima de src)
BASE_DIR = Path(__file__).resolve().parent.parent
# Banco vetorial
CHROMA_DIR = BASE_DIR / "data" / "vector_store" / "chroma"

def verificar_banco():

    print_subsection("\n========== VERIFICAÇÃO DO CHROMADB ==========")
    print_info(f"Caminho do banco: {CHROMA_DIR}")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    colecoes = client.list_collections()

    print_info(f"Total de coleções: {len(colecoes)}")

    for colecao in colecoes:

        print(f"\n========== COLEÇÃO: {colecao.name} ==========")

        dados = colecao.get(
            include=["documents", "metadatas"]
        )

        print(f"Total de registros: {len(dados['ids'])}")

        for i, id_documento in enumerate(dados["ids"], start=1):

            conteudo = dados["documents"][i - 1]
            metadados = dados["metadatas"][i - 1]

            print(f"\n--- Registro {i} ---")
            print(f"ID: {id_documento}")
            print(f"Caracteres: {len(conteudo)}")
            print(f"Metadados: {metadados}")
            print(f"Conteúdo:\n{conteudo}")

    print("\n========== FIM DA VERIFICAÇÃO ==========")

if __name__ == "__main__":
    verificar_banco()
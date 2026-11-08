# Local RAG — Sistema de Perguntas e Respostas sobre Documentos PDF

Um sistema de Retrieval-Augmented Generation (RAG) desenvolvido em Python para consultar documentos PDF utilizando inteligência artificial local.

A aplicação permite fazer perguntas em linguagem natural sobre documentos pessoais e profissionais, recuperando trechos relevantes por similaridade semântica e utilizando um modelo de linguagem para elaborar respostas contextualizadas.

O projeto utiliza Ollama para executar os modelos localmente, ChromaDB para armazenamento e recuperação vetorial e LangChain para integrar as diferentes etapas do processamento.

O principal objetivo é explorar, na prática, como construir uma aplicação baseada em RAG, entendendo não apenas como conectar um modelo de linguagem a documentos, mas também como funciona o processo de extração, fragmentação, indexação e recuperação de informações.

## Objetivos do projeto

O desenvolvimento deste projeto tem como foco:

- Construir um pipeline completo de RAG utilizando ferramentas open source.
- Processar documentos PDF e transformar seu conteúdo em unidades menores de informação (chunks).
- Gerar embeddings para representar semanticamente os textos.
- Armazenar e consultar vetores utilizando um banco de dados vetorial.
- Recuperar informações relevantes a partir de perguntas em linguagem natural.
- Integrar um LLM local para gerar respostas baseadas no contexto recuperado.
- Preservar a rastreabilidade das informações, identificando os documentos e páginas utilizados como fontes.

## Tecnologias utilizadas

| Tecnologia | Responsabilidade |
|---|---|
| Python 3.12 | Linguagem principal |
| LangChain | Orquestração do pipeline RAG |
| Ollama | Execução local dos modelos de IA |
| Qwen 3.5 4B | Modelo de linguagem para geração de respostas |
| Nomic Embed Text | Modelo utilizado para geração de embeddings |
| ChromaDB | Armazenamento e busca vetorial |
| PyPDFLoader | Extração de conteúdo dos documentos PDF |
| RecursiveCharacterTextSplitter | Fragmentação dos documentos |

## Como funciona

O sistema é organizado em etapas independentes, permitindo entender e acompanhar o caminho percorrido desde a entrada dos documentos até a geração da resposta.

### 1. Carregamento dos documentos

Os arquivos PDF são armazenados no diretório `docs/`.

O carregador percorre os documentos disponíveis e extrai seu conteúdo, mantendo informações de metadados, como origem do arquivo e número da página.

Essa etapa utiliza o `PyPDFLoader`, disponibilizado pelo ecossistema LangChain.

### 2. Fragmentação do conteúdo (Chunking)

Após a extração, os documentos são divididos em fragmentos menores, chamados de chunks.

Essa divisão é necessária porque trabalhar com documentos inteiros pode dificultar a recuperação de informações específicas e aumentar a quantidade de contexto enviada ao modelo.

O projeto utiliza `RecursiveCharacterTextSplitter`, com os seguintes parâmetros iniciais:

- Chunk size: 500 caracteres.
- Chunk overlap: 100 caracteres.

O overlap permite que fragmentos consecutivos compartilhem parte do conteúdo, reduzindo a perda de contexto nas regiões em que ocorre a divisão.

### 3. Geração de embeddings

Cada chunk é convertido em uma representação vetorial utilizando o modelo `nomic-embed-text`.

Os embeddings permitem representar semanticamente os textos, possibilitando encontrar conteúdos relacionados à pergunta mesmo quando as palavras utilizadas não são exatamente iguais às presentes no documento.

### 4. Armazenamento vetorial

Os vetores gerados são armazenados no ChromaDB.

O banco mantém os embeddings associados aos respectivos conteúdos e metadados, permitindo realizar consultas semânticas posteriormente.

O armazenamento persistente evita a necessidade de reconstruir toda a base vetorial a cada execução, desde que os documentos não tenham sido alterados.

### 5. Recuperação de informações (Retrieval)

Quando uma pergunta é enviada, ela também é transformada em embedding.

O Retriever utiliza essa representação para pesquisar os chunks semanticamente mais próximos dentro da base vetorial.

Os documentos recuperados são utilizados como contexto para a próxima etapa.

### 6. Geração da resposta (Generation)

O contexto recuperado é encaminhado ao modelo de linguagem `qwen3.5:4b`, executado localmente pelo Ollama.

O modelo analisa as informações disponíveis e gera uma resposta em linguagem natural.

Além da resposta, a aplicação retorna os documentos utilizados como fontes, permitindo identificar de onde vieram as informações apresentadas.

## Arquitetura do sistema

O fluxo principal da aplicação pode ser representado da seguinte maneira:

```text
             DOCUMENTOS PDF
                    |
                    v
              PDF LOADER
                    |
                    v
              TEXT SPLITTER
                    |
                    v
                CHUNKS
                    |
                    v
                EMBEDDINGS
                    |
                    v
               CHROMADB
                    |
                    |
       +------------+-------------+
       |                          |
       v                          v
  PERGUNTA DO USUÁRIO         BASE VETORIAL
       |                          |
       v                          |
  QUERY EMBEDDING                 |
       |                          |
       +----------+---------------+
                  |
                  v
              RETRIEVER
                  |
                  v
          CONTEXTO RECUPERADO
                  |
                  v
             OLLAMA / LLM
                  |
                  v
          RESPOSTA + FONTES
```

## Estrutura do projeto

O código está organizado dentro do diretório `src/`, separando as responsabilidades de configuração, carregamento de documentos, geração de embeddings, armazenamento vetorial e construção da cadeia de perguntas e respostas.

```text
rag/
│
├── src/
│   ├── docs/
│   │   ├── arquivo1.pdf
│   │   └── arquivo2.pd
│   │
│   ├── config.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── qa_chain.py
│   └── vector_store.py
│
├── chroma_cover_letter/
│
├── .venv/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Responsabilidade dos módulos

| Arquivo | Responsabilidade |
|---|---|
| `config.py` | Centraliza configurações e parâmetros utilizados pela aplicação. |
| `document_loader.py` | Localiza os arquivos PDF e realiza a extração e o processamento inicial dos documentos. |
| `embeddings.py` | Configura e disponibiliza o modelo responsável pela geração de embeddings. |
| `vector_store.py` | Gerencia a integração com o ChromaDB e o armazenamento vetorial dos documentos. |
| `qa_chain.py` | Configura o modelo de linguagem e constrói a cadeia de perguntas e respostas utilizando o Retriever. |
| `main.py` | Ponto de entrada da aplicação, responsável por iniciar e orquestrar o fluxo principal. |

Essa separação permite manter cada etapa do pipeline independente, facilitando a manutenção, os testes e a evolução do sistema.

## Requisitos

Antes de executar o projeto, é necessário ter:

- Python 3.12 ou superior.
- Ollama instalado.
- Git (opcional, para clonar o repositório).
- Modelos necessários disponíveis no Ollama.

## Configuração do ambiente

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git

cd rag
```

### 2. Criar o ambiente virtual

```bash
python -m venv .venv
```

Ativar no Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar os modelos no Ollama

Modelo de linguagem:

```bash
ollama pull qwen3.5:4b
```

Modelo de embeddings:

```bash
ollama pull nomic-embed-text
```

Certifique-se de que o Ollama esteja em execução antes de iniciar a aplicação.

### 5. Executar o projeto

```bash
python main.py
```

Os documentos disponíveis no diretório configurado serão processados e utilizados para alimentar a base vetorial, conforme a implementação atual da aplicação.

## Exemplo de utilização

Considere um conjunto de documentos contendo currículo, histórico profissional e carta de apresentação.

Uma pergunta possível:

```text
Qual foi o primeiro emprego de Gabriel Sampaio?
```

O sistema deve:

1. Converter a pergunta em embedding.
2. Consultar a base vetorial.
3. Recuperar os trechos relacionados ao histórico profissional.
4. Encaminhar o contexto ao modelo de linguagem.
5. Gerar uma resposta fundamentada nos documentos recuperados.

Exemplo de resposta esperada:

```text
A primeira experiência profissional registrada
foi como estagiário, entre outubro de 2019
e janeiro de 2020.
```

A aplicação também apresenta as fontes utilizadas na resposta, incluindo o nome do arquivo e a página de origem.

## Decisões técnicas e aprendizados

Durante o desenvolvimento, alguns aspectos importantes do funcionamento de um sistema RAG estão sendo explorados:

**Fragmentação e preservação de contexto**

O tamanho dos chunks influencia diretamente a recuperação de informações. Fragmentos muito pequenos podem perder contexto, enquanto fragmentos muito grandes podem trazer informações pouco relevantes para a pergunta.

**Busca semântica**

A recuperação não depende exclusivamente da correspondência exata entre palavras. Os embeddings permitem encontrar relações de significado entre perguntas e documentos.

**Qualidade da recuperação**

Um LLM não consegue responder corretamente com informações que não foram recuperadas ou que não estão presentes no contexto fornecido. Por isso, avaliar os resultados do Retriever é tão importante quanto avaliar a resposta final.

**Rastreabilidade**

Retornar as fontes permite verificar se a resposta realmente está fundamentada nos documentos originais.

**Execução local**

A utilização do Ollama permite executar tanto a geração de respostas quanto a criação de embeddings localmente, reduzindo a dependência de serviços externos e APIs pagas.

## Possíveis melhorias

O projeto está sendo desenvolvido de forma incremental, com foco em compreender e melhorar cada etapa do pipeline.

Algumas evoluções planejadas:

- Implementar estratégias de chunking que preservem melhor a estrutura dos documentos.
- Comparar diferentes tamanhos de chunks e valores de overlap.
- Avaliar a qualidade da recuperação utilizando métricas como Precision@K e Recall@K.
- Comparar busca vetorial tradicional com estratégias híbridas (BM25 + busca semântica).
- Implementar reranking para melhorar a relevância dos documentos recuperados.
- Melhorar a identificação e a apresentação das fontes.
- Implementar mecanismos para atualização incremental da base vetorial.
- Desenvolver testes automatizados para as etapas de processamento e recuperação.

## Considerações finais

Este projeto surgiu como uma forma de estudar e aplicar conceitos de inteligência artificial generativa além da simples utilização de APIs de modelos de linguagem.

A proposta é entender como as informações são extraídas, representadas matematicamente, armazenadas e recuperadas antes de chegar ao modelo responsável pela geração da resposta.

Mais do que produzir respostas, o objetivo é construir um sistema em que seja possível investigar o caminho percorrido pela informação, identificar limitações e melhorar progressivamente a qualidade dos resultados.

---

**Desenvolvido por Gabriel Sampaio**

[GitHub](https://github.com/GabrielSampaioS) · [LinkedIn](https://www.linkedin.com/in/gabriel-sampaio0122/)
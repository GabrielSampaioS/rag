"""
metadata_classifier.py

Classificador de metadados (baseado em regras, sem LLM) para chunks de
currículo (pt) e carta de apresentação (en).

Principais diferenças em relação à versão anterior:
  1. Pontuação ponderada (cabeçalho > padrão estrutural > palavra-chave),
     em vez de "achou uma palavra = achou a seção".
  2. Multi-rótulo: um chunk de 500 caracteres costuma misturar seções.
     Guardamos uma categoria principal + flags has_<categoria> filtráveis.
  3. Herança: chunk sem sinal forte herda a categoria do chunk anterior
     (os cabeçalhos ficam em um chunk e o conteúdo cai nos seguintes).
  4. Sinais estruturais: intervalos de datas, bullets com verbos de ação,
     e-mail/telefone/URLs, "Banco de Dados:", "Inglês: Intermediário"...
  5. Limpeza de ruído do PDF (cirílico do LinkedIn, "Page 1 of 3", bullets).
  6. Metadados no formato aceito pelo ChromaDB (str/int/float/bool).
  7. Lado da pergunta: classify_query + filtro suave no retriever.
"""

import re
import unicodedata
from functools import lru_cache

# ============================================================
# CONFIGURAÇÃO EDITÁVEL
# ============================================================

COMPANIES = ["boersoft", "glp laboratorios", "clima core"]
PROJECT_NAMES = [
    "ledgerflow", "cinema reservation system",
    "asset batch manager", "local rag",
]

MIN_SCORE = 4            # abaixo disso o chunk herda a categoria anterior
SECONDARY_RATIO = 0.4    # categoria secundária: >= 40% da pontuação da principal

# ============================================================
# NORMALIZAÇÃO E LIMPEZA
# ============================================================


def clean_text(text: str) -> str:
    """Remove ruídos típicos de extração de PDF, mantendo maiúsculas/acentos."""
    text = re.sub(r"[\u0400-\u04FF]+", " ", text)      # ícone de local do LinkedIn vira cirílico
    text = re.sub(r"Page \d+ of \d+", " ", text)       # rodapé do LinkedIn
    text = re.sub(r"[●•▪■◦]", " • ", text)             # uniformiza bullets
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = clean_text(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=None)
def _term_re(term: str):
    return re.compile(rf"(?<!\w){re.escape(normalize_text(term))}(?!\w)")


def _has(term: str, norm: str) -> bool:
    return _term_re(term).search(norm) is not None


# ============================================================
# DATAS (sinal forte de "experiência" ou "formação")
# ============================================================

MONTHS_PT = "janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
MONTHS_EN = "january|february|march|april|may|june|july|august|september|october|november|december"
_DATE = rf"(?:\d{{2}}/\d{{4}}|(?:{MONTHS_PT}) de \d{{4}}|(?:{MONTHS_EN}) \d{{4}})"
DATE_RANGE = rf"{_DATE}\s*[-–]\s*(?:{_DATE}|atual|atualmente|presente|present|current)"

ACTION_BULLET = (
    r"•\s*(?:desenvolvimento|elaboracao|participacao|validacao|atendimento|"
    r"colaboracao|padronizacao|execucao|modelagem|implantacao|manutencao|"
    r"criacao|gestao|analise|apoio|interface|suporte|responsavel|atuacao|"
    r"integracao|automacao|levantamento|revisao|controle|parametrizacao|calibracao)\b"
)

# ============================================================
# REGRAS DE CATEGORIA
#   headings_raw : regex CASE-SENSITIVE no texto limpo (peso 5)
#   phrases      : frases normalizadas inequívocas (peso 3, máx. 2)
#   names        : nomes próprios configuráveis (peso 2, máx. 2)
#   keywords     : palavras soltas (peso 1, máx. 5)
#   patterns     : (regex normalizado, peso, máx. de ocorrências)
# ============================================================

RULES = {
    "professional_profile": {
        "headings_raw": [
            r"(?:^|\s)(?:Resumo|RESUMO|Sobre mim|SOBRE MIM|Perfil [Pp]rofissional|"
            r"PERFIL PROFISSIONAL|Objetivo(?: [Pp]rofissional)?|OBJETIVO|"
            r"(?:Professional )?Summary|SUMMARY|About [Mm]e)(?=\s|:|$)"
        ],
        "phrases": [
            "sobre mim", "perfil profissional", "objetivo profissional",
            "professional summary", "career objective", "about me",
            "busco oportunidades", "tenho interesse", "atualmente curso",
            "i am writing to", "i am applying", "i am excited",
            "i am passionate", "i would like to", "i would love to",
            "looking for an opportunity", "dear hiring", "sincerely",
            "thank you for",
        ],
        "names": [],
        "keywords": [],
        "patterns": [],
    },
    "professional_experience": {
        "headings_raw": [
            r"(?:^|\s)(?:Experiência(?: [Pp]rofissional)?|EXPERIÊNCIA(?: PROFISSIONAL)?|"
            r"Histórico [Pp]rofissional|(?:Work|Professional) [Ee]xperience|"
            r"Employment [Hh]istory)(?=\s+[A-ZÀ-Ý]|:|$)"
        ],
        "phrases": [
            "experiencia profissional", "historico profissional",
            "work experience", "professional experience", "employment history",
            "atividades desenvolvidas", "principais atividades",
            "in my role", "in my previous role", "in my current role",
        ],
        "names": COMPANIES,
        "keywords": [
            "assistente", "tecnico de laboratorio", "estagiario", "auxiliar",
            "analista", "estagio", "coordenador", "responsabilidades",
            "conquistas", "achievements", "responsibilities",
        ],
        "patterns": [
            (DATE_RANGE, 2, 3),
            (ACTION_BULLET, 2, 3),
            (r"\(\d+ (?:anos?|mes(?:es)?)(?: \d+ mes(?:es)?)?\)", 2, 2),
            (r"\b(?:i|we) (?:developed|built|led|worked|maintained|implemented|"
             r"supported|collaborated|designed|created|managed|automated|participated)\b", 2, 3),
        ],
    },
    "education": {
        "headings_raw": [
            r"(?:^|\s)(?:Formação(?: [Aa]cadêmica)?|FORMAÇÃO(?: ACADÊMICA)?|Educação|"
            r"EDUCAÇÃO|Education|EDUCATION|Certifications|Certificações|Certificados|"
            r"CERTIFICAÇÕES|Cursos [Cc]omplementares)(?=\s|:|$)"
        ],
        "phrases": [
            "formacao academica", "ensino superior", "curso tecnico",
            "formacao tecnica", "academic education", "higher education",
            "technical education", "cursos complementares",
            "certificado de participacao",
        ],
        "names": [],
        "keywords": [
            "bacharelado", "graduacao", "univesp", "etec", "universidade",
            "faculdade", "bachelor", "degree", "university", "certificado",
            "certificacao", "engenharia da computacao", "cursando",
        ],
        "patterns": [
            (r"\b(?:bacharelado|tecnico|tecnologo|bachelor|mba|pos[- ]graduacao) "
             r"(?:em|of|in) ", 3, 2),
        ],
    },
    "technical_skills": {
        "headings_raw": [
            r"(?:^|\s)(?:Competências|COMPETÊNCIAS|Principais competências|"
            r"Principais tecnologias|Habilidades|HABILIDADES|Skills|SKILLS|"
            r"Top Skills|Tecnologias|TECNOLOGIAS)(?=\s|:|$)"
        ],
        "phrases": [
            "principais tecnologias", "principais competencias",
            "programming languages", "linguagens de programacao",
            "technical skills", "hard skills",
        ],
        "names": [],
        "keywords": [],
        "patterns": [
            (r"\b(?:banco de dados|ferramentas|tecnologias|sistemas|linguagens|"
             r"frameworks|suporte tecnico|metodologias)\s*:", 2, 3),
        ],
        # + densidade de tecnologias detectadas (ver generate_metadata)
    },
    "languages": {
        "headings_raw": [
            r"(?:^|\s)(?:Idiomas|IDIOMAS|Idioma|(?<!Programming )Languages|LANGUAGES)(?=\s|:|$)"
        ],
        "phrases": [],
        "names": [],
        "keywords": [],
        "patterns": [
            (r"\b(?:ingles|espanhol|portugues|frances|alemao|english|spanish|"
             r"portuguese|french|german)\s*:\s*(?:basico|intermediario|avancado|"
             r"fluente|nativo|basic|intermediate|advanced|fluent|native)", 5, 2),
        ],
    },
    "contact_information": {
        "headings_raw": [
            r"(?:^|\s)(?:Contato|CONTATO|Contact(?: [Ii]nformation)?|CONTACT)(?=\s|:|$)"
        ],
        "phrases": [],
        "names": [],
        "keywords": [],
        "patterns": [
            (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*", 4, 2),
            (r"\b\d{10,11}\b", 3, 1),
            (r"\(\d{2}\)\s?9?\d{4}-?\d{4}", 3, 1),
            (r"(?:linkedin\.com|github\.com|leetcode\.com)", 3, 2),
            (r"\((?:mobile|linkedin|portfolio|email)\)", 2, 2),
        ],
    },
    "projects": {
        "headings_raw": [
            r"(?:^|\s)(?:Projetos(?: [Pp]essoais)?|PROJETOS|Projects|PROJECTS|"
            r"Personal Projects)(?=\s|:|$)"
        ],
        "phrases": [
            "projetos pessoais", "personal projects", "side projects",
            "projetos academicos", "academic projects", "projetos profissionais",
        ],
        "names": PROJECT_NAMES,
        "keywords": [],
        "patterns": [],
    },
}

SECTION_LABELS = {
    "professional_profile": "Resumo / perfil profissional",
    "professional_experience": "Experiência profissional",
    "education": "Formação e certificações",
    "technical_skills": "Competências técnicas",
    "languages": "Idiomas",
    "contact_information": "Contato",
    "projects": "Projetos",
    "unclassified": "Não classificada",
}

DOC_LABELS = {"resume": "Currículo", "cover_letter": "Carta de apresentação"}

# ============================================================
# TÓPICOS
# ============================================================

TOPIC_RULES = {
    "python": ["python"],
    "javascript": ["javascript"],
    "typescript": ["typescript"],
    "nodejs": ["node.js", "nodejs"],
    "express": ["express", "express.js"],
    "rest_api": ["rest api", "rest apis", "restful api", "apis rest", "api rest"],
    "sql": ["sql", "structured query language", "consultas sql"],
    "sql_server": ["sql server"],
    "stored_procedures": ["stored procedures", "procedures"],
    "delphi": ["delphi"],
    "react": ["react"],
    "flutter": ["flutter"],
    "dart": ["dart"],
    "langchain": ["langchain"],
    "chromadb": ["chromadb", "chroma db"],
    "ollama": ["ollama"],
    "docker": ["docker"],
    "git": ["git", "github"],
    "github_actions": ["github actions"],
    "ci_cd": ["ci/cd", "integracao continua", "continuous integration"],
    "linux": ["linux"],
    "aws": ["aws", "amazon web services"],
    "elasticsearch": ["elasticsearch"],
    "mongodb": ["mongodb"],
    "redis": ["redis"],
    "postgresql": ["postgresql"],
    "vba": ["vba"],
    "excel": ["excel"],
    "erp": ["erp"],
    "devops": ["devops"],
    "automated_testing": [
        "testes automatizados", "testes unitarios", "testes de integracao",
        "e2e", "automated testing", "unit tests",
    ],
    "computer_engineering": ["engenharia da computacao", "computer engineering"],
    "technical_support": ["suporte tecnico", "technical support", "troubleshooting", "service desk"],
    "system_implementation": [
        "implantacao de sistemas", "implementacao de sistemas",
        "system implementation", "implantacao",
    ],
    "requirements_gathering": [
        "levantamento de requisitos", "requirements gathering", "requirements analysis",
    ],
    "software_development": ["desenvolvimento de software", "software development"],
    "artificial_intelligence": [
        "inteligencia artificial", "artificial intelligence", "machine learning",
    ],
    "data_analysis": ["analise de dados", "data analysis"],
    "cloud_computing": ["computacao em nuvem", "cloud computing"],
}

TECH_TOPICS = {
    "python", "javascript", "typescript", "nodejs", "express", "rest_api",
    "sql", "sql_server", "stored_procedures", "delphi", "react", "flutter",
    "dart", "langchain", "chromadb", "ollama", "docker", "git",
    "github_actions", "ci_cd", "linux", "aws", "elasticsearch", "mongodb",
    "redis", "postgresql", "vba", "excel", "erp", "devops", "automated_testing",
}


def detect_topics(norm: str) -> list:
    return [
        topic for topic, terms in TOPIC_RULES.items()
        if any(_has(t, norm) for t in terms)
    ]


# ============================================================
# CLASSIFICAÇÃO DO CHUNK
# ============================================================


def _score(rule: dict, raw: str, norm: str) -> int:
    score = 0

    for pattern in rule["headings_raw"]:
        if re.search(pattern, raw):
            score += 5

    score += 3 * min(sum(_has(p, norm) for p in rule["phrases"]), 2)
    score += 2 * min(sum(_has(n, norm) for n in rule["names"]), 2)
    score += min(sum(_has(k, norm) for k in rule["keywords"]), 5)

    for pattern, weight, cap in rule["patterns"]:
        score += weight * min(len(re.findall(pattern, norm)), cap)

    return score


def generate_metadata(text: str, previous_category: str | None = None) -> dict:
    """
    Retorna metadados compatíveis com ChromaDB para um chunk.

    previous_category: categoria do chunk anterior do mesmo documento,
    usada quando o chunk atual não tem sinais suficientes.
    """
    raw = clean_text(text)
    norm = normalize_text(text)

    topics = detect_topics(norm)
    scores = {cat: _score(rule, raw, norm) for cat, rule in RULES.items()}

    # Densidade de tecnologias reforça "competências técnicas"
    n_tech = sum(t in TECH_TOPICS for t in topics)
    scores["technical_skills"] += min(n_tech, 6)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_cat, top_score = ranked[0]

    inherited = False
    if top_score < MIN_SCORE:
        if previous_category and previous_category != "unclassified":
            top_cat, inherited = previous_category, True
            categories = [top_cat]
        else:
            top_cat, categories = "unclassified", []
    else:
        threshold = max(MIN_SCORE, SECONDARY_RATIO * top_score)
        categories = [c for c, s in ranked if s >= threshold]

    metadata = {
        "category": top_cat,
        "sections": ",".join(categories) or "unclassified",
        "topics": ",".join(topics) or "unclassified",
        "category_score": top_score,
        "category_inherited": inherited,
    }
    # Flags booleanas: o Chroma não filtra por "contém" em strings,
    # mas filtra por igualdade em booleanos.
    for cat in RULES:
        metadata[f"has_{cat}"] = cat in categories

    return metadata


# ============================================================
# NÍVEL DO DOCUMENTO
# ============================================================

_PT = re.compile(r"\b(?:de|da|do|das|dos|em|para|com|que|uma|um|os|as|na|no|por|ao)\b")
_EN = re.compile(r"\b(?:the|and|of|to|in|with|for|that|is|my|are|was|have|at|on)\b")


def detect_language(text: str) -> str:
    norm = normalize_text(text)
    return "pt" if len(_PT.findall(norm)) >= len(_EN.findall(norm)) else "en"


def detect_doc_type(source: str) -> str:
    filename = normalize_text(re.split(r"[\\/]", source)[-1])
    if any(k in filename for k in ("cover", "carta")):
        return "cover_letter"
    return "resume"


def build_header(metadata: dict) -> str:
    doc = DOC_LABELS.get(metadata.get("doc_type"), "Documento")
    section = SECTION_LABELS.get(metadata.get("category"), "Não classificada")
    return f"[Documento: {doc} | Seção: {section}]"


def enrich_chunks(chunks: list, prepend_header: bool = True) -> list:
    """
    Recebe os chunks (Documents do LangChain) NA ORDEM em que o splitter
    os gerou e preenche os metadados de cada um.

    prepend_header: adiciona "[Documento: ... | Seção: ...]" no início do
    texto. Assim o contexto entra no embedding e o retriever diferencia,
    por exemplo, "Técnico de Laboratório" (experiência) de "Técnico em
    Desenvolvimento de Sistemas" (formação).
    """
    texts_by_source: dict = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        texts_by_source.setdefault(source, []).append(chunk.page_content)

    language_by_source = {
        src: detect_language(" ".join(texts))
        for src, texts in texts_by_source.items()
    }

    previous: dict = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        meta = generate_metadata(chunk.page_content, previous.get(source))

        if meta["category"] != "unclassified":
            previous[source] = meta["category"]

        meta["doc_type"] = detect_doc_type(source)
        meta["language"] = language_by_source[source]
        chunk.metadata.update(meta)

        if prepend_header and not chunk.page_content.startswith("[Documento:"):
            chunk.page_content = f"{build_header(chunk.metadata)}\n{chunk.page_content}"

    return chunks


# ============================================================
# LADO DA PERGUNTA
# ============================================================

QUERY_HINTS = {
    "professional_experience": [
        "emprego", "trabalh", "experiencia", "empresa", "cargo", "estagi",
        "carreira", "atuou", "atuacao", "job", "worked",
    ],
    "education": [
        "formacao", "estud", "faculdade", "curso", "graduacao",
        "universidade", "diploma", "certificad", "formou", "degree",
    ],
    "technical_skills": [
        "tecnologia", "linguagem", "habilidade", "competencia", "stack",
        "ferramenta", "dominio", "sabe usar", "skills",
    ],
    "languages": ["ingles", "idioma", "fluente", "espanhol", "fluencia"],
    "contact_information": [
        "email", "e-mail", "telefone", "contato", "linkedin", "github",
        "celular", "cidade",
    ],
    "professional_profile": ["resumo", "objetivo", "perfil", "sobre voce", "apresent"],
    "projects": ["projeto", "project"],
}


def classify_query(question: str) -> list:
    """Categorias prováveis da pergunta, da mais para a menos provável."""
    q = normalize_text(question)
    hits = {
        cat: sum(stem in q for stem in stems)
        for cat, stems in QUERY_HINTS.items()
    }
    return [c for c, n in sorted(hits.items(), key=lambda kv: -kv[1]) if n > 0]


def build_chroma_filter(categories: list):
    flags = [{f"has_{c}": True} for c in categories]
    if not flags:
        return None
    return flags[0] if len(flags) == 1 else {"$or": flags}


def retrieve_with_filter(vector_store, question: str, k: int = 6) -> list:
    """
    Filtro SUAVE: busca primeiro nas categorias prováveis e completa com a
    busca sem filtro, para que um erro de classificação nunca esconda a
    resposta.
    """
    flt = build_chroma_filter(classify_query(question))
    docs = vector_store.similarity_search(question, k=k, filter=flt) if flt else []

    if len(docs) < k:
        seen = {d.page_content for d in docs}
        for doc in vector_store.similarity_search(question, k=k):
            if doc.page_content not in seen:
                docs.append(doc)
                seen.add(doc.page_content)
            if len(docs) >= k:
                break
    return docs

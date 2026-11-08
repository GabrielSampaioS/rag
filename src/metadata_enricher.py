
import re
import unicodedata


# ==========================================
# REGRAS DE CLASSIFICAÇÃO MANUAL
# ==========================================

SECTION_RULES = {
    "personal_presentation": {
        "category": "professional_profile",
        "keywords": [
            "sobre mim", "perfil profissional",
            "about me", "professional profile",
            "professional summary", "summary",
            "career summary"
        ]
    },

    "career_objective": {
        "category": "professional_profile",
        "keywords": [
            "objetivo profissional", "objetivo",
            "career objective", "professional objective",
            "seeking a position", "looking for an opportunity"
        ]
    },

    "work_experience": {
        "category": "professional_experience",
        "keywords": [
            "experiencia profissional", "experiencia de trabalho",
            "historico profissional", "historico de emprego",
            "work experience", "professional experience",
            "employment history", "work history"
        ]
    },

    "responsibilities": {
        "category": "professional_experience",
        "keywords": [
            "responsabilidades", "atividades desenvolvidas",
            "principais atividades", "responsibilities",
            "main duties", "job duties"
        ]
    },

    "achievements": {
        "category": "professional_experience",
        "keywords": [
            "conquistas", "resultados alcancados",
            "realizacoes", "achievements",
            "accomplishments", "results"
        ]
    },

    "academic_education": {
        "category": "education",
        "keywords": [
            "formacao academica", "ensino superior",
            "graduacao", "bacharelado",
            "academic education", "higher education",
            "bachelor", "university", "degree"
        ]
    },

    "technical_education": {
        "category": "education",
        "keywords": [
            "formacao tecnica", "curso tecnico",
            "technical education", "technical training",
            "vocational training"
        ]
    },

    "certifications": {
        "category": "education",
        "keywords": [
            "certificados", "certificacoes",
            "cursos complementares", "certifications",
            "certificates", "additional courses"
        ]
    },

    "programming_languages": {
        "category": "technical_skills",
        "keywords": [
            "python", "javascript", "typescript",
            "java", "c#", "c++", "delphi",
            "programming languages", "linguagens de programacao"
        ]
    },

    "frameworks": {
        "category": "technical_skills",
        "keywords": [
            "node.js", "nodejs", "express",
            "react", "langchain", "fastapi",
            "frameworks", "bibliotecas"
        ]
    },

    "databases": {
        "category": "technical_skills",
        "keywords": [
            "postgresql", "mongodb", "mysql",
            "sqlite", "sql server", "chromadb",
            "bancos de dados", "databases"
        ]
    },

    "tools": {
        "category": "technical_skills",
        "keywords": [
            "docker", "git", "github", "linux",
            "ollama", "aws", "elasticsearch",
            "ferramentas", "tools"
        ]
    },

    "personal_projects": {
        "category": "projects",
        "keywords": [
            "projetos pessoais", "personal projects",
            "side projects"
        ]
    },

    "academic_projects": {
        "category": "projects",
        "keywords": [
            "projetos academicos", "projetos universitarios",
            "academic projects", "university projects"
        ]
    },

    "professional_projects": {
        "category": "projects",
        "keywords": [
            "projetos profissionais", "professional projects",
            "projects developed at work"
        ]
    },

    "contact_information": {
        "category": "contact_information",
        "keywords": [
            "contato", "informacoes de contato",
            "contact", "contact information",
            "email", "telefone", "phone",
            "linkedin", "github", "portfolio"
        ]
    }
}


# ==========================================
# TÓPICOS TÉCNICOS E PROFISSIONAIS
# ==========================================

TOPIC_RULES = {
    "python": ["python"],
    "javascript": ["javascript"],
    "typescript": ["typescript"],
    "nodejs": ["node.js", "nodejs"],
    "rest_api": ["rest api", "restful api", "apis rest"],
    "sql": ["sql", "structured query language"],
    "delphi": ["delphi"],
    "react": ["react"],
    "langchain": ["langchain"],
    "chromadb": ["chromadb", "chroma db"],
    "docker": ["docker"],
    "git": ["git", "github"],
    "linux": ["linux"],
    "aws": ["aws", "amazon web services"],
    "elasticsearch": ["elasticsearch"],
    "mongodb": ["mongodb"],
    "postgresql": ["postgresql"],
    "computer_engineering": [
        "engenharia da computacao",
        "computer engineering"
    ],
    "technical_support": [
        "suporte tecnico", "technical support",
        "troubleshooting"
    ],
    "system_implementation": [
        "implantacao de sistemas",
        "implementacao de sistemas",
        "system implementation"
    ],
    "requirements_gathering": [
        "levantamento de requisitos",
        "requirements gathering",
        "requirements analysis"
    ],
    "software_development": [
        "desenvolvimento de software",
        "software development"
    ],
    "artificial_intelligence": [
        "inteligencia artificial",
        "artificial intelligence",
        "machine learning"
    ],
    "data_analysis": [
        "analise de dados",
        "data analysis"
    ],
    "cloud_computing": [
        "computacao em nuvem",
        "cloud computing"
    ]
}


# ==========================================
# NORMALIZAÇÃO DE TEXTO
# ==========================================

def normalize_text(text: str) -> str:
    text = text.lower()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", text).strip()


# ==========================================
# GERAÇÃO MANUAL DE METADADOS
# ==========================================

def generate_metadata(text: str) -> dict:
    """
    Gera metadados semânticos utilizando regras manuais.

    Futuramente, esta função poderá ser substituída
    por uma implementação baseada em LLM.
    """

    normalized_text = normalize_text(text)

    matched_sections = []
    matched_topics = []

    # Identifica seções e categorias
    for section, rules in SECTION_RULES.items():

        for keyword in rules["keywords"]:
            keyword_normalized = normalize_text(keyword)

            if re.search(
                rf"(?<!\w){re.escape(keyword_normalized)}(?!\w)",
                normalized_text
            ):
                matched_sections.append(section)
                break

    # Identifica tópicos
    for topic, keywords in TOPIC_RULES.items():

        for keyword in keywords:
            keyword_normalized = normalize_text(keyword)

            if re.search(
                rf"(?<!\w){re.escape(keyword_normalized)}(?!\w)",
                normalized_text
            ):
                matched_topics.append(topic)
                break

    # Identifica a categoria predominante
    category_counts = {}

    for section in matched_sections:
        category = SECTION_RULES[section]["category"]

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    category = (
        max(category_counts, key=category_counts.get)
        if category_counts else "unclassified"
    )

    return {
        "category": category,
        "sections": ",".join(matched_sections) or "unclassified",
        "topics": ",".join(matched_topics) or "unclassified"
    }

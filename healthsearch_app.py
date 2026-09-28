"""
HealthSearch - Motor de Busca Hibrido (BM25 + Busca Semantica + RRF)
UNIPE - Tendencias em Ciencia da Computacao (Recuperacao de Informacao)

Como rodar:
    pip install streamlit pandas rank_bm25 sentence-transformers
    streamlit run healthsearch_app.py

Obs: se o sentence-transformers (ou o modelo) nao estiver disponivel, o app
usa uma SIMULACAO VETORIAL documentada (ver secao da Fase 3), entao ele roda
mesmo sem internet.
"""

import re
import time
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    TEM_SENTENCE_TRANSFORMERS = True
except ImportError:
    TEM_SENTENCE_TRANSFORMERS = False


# =====================================================================
# FASE 1 - Corpus medico (hardcode) e pre-processamento
# =====================================================================

CORPUS = [
    {"id": "Doc 1", "titulo": "Protocolo Emergência ECG",
     "conteudo": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana "
                 "devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."},
    {"id": "Doc 2", "titulo": "Guia de Farmacologia Cardíaca",
     "conteudo": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários "
                 "reduz a mortalidade no infarto agudo do miocárdio."},
    {"id": "Doc 3", "titulo": "Diretriz de Hipertensão Arterial",
     "conteudo": "A crise hipertensiva severa requer administração de anti-hipertensivos "
                 "venosos e monitoramento contínuo da pressão arterial na UTI."},
    {"id": "Doc 4", "titulo": "Manual de AVC Isquêmico",
     "conteudo": "O acidente vascular cerebral isquêmico agudo deve ser tratado com "
                 "trombolíticos venosos em até quatro horas e meia do início dos sintomas."},
    {"id": "Doc 5", "titulo": "Protocolo de Reanimação RCR",
     "conteudo": "Parada cardiorrespiratória em adultos exige compressões torácicas "
                 "contínuas de alta qualidade e desfibrilação precoce no código azul."},
    {"id": "Doc 6", "titulo": "Procedimentos de UTI Geral",
     "conteudo": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, "
                 "recomenda-se a monitorização cardíaca contínua por telemetria."},
]

# stopwords em portugues (ja sem acento, porque a limpeza tira os acentos antes)
STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "pelo", "pela", "com", "sem", "sobre",
    "entre", "ate", "apos", "e", "ou", "mas", "que", "se", "ao", "aos", "como", "quando",
    "ser", "foi", "sao", "ha", "mais", "muito", "seu", "sua", "este", "esta", "isso",
    "ele", "ela", "ja", "tambem",
}


def remover_acentos(texto):
    """'síndrome' -> 'sindrome' (decompoe o caractere e joga fora o acento)."""
    decomposto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn")


def limpar_texto(texto):
    """Minusculas + sem acento + sem caracteres especiais. Ex: 'CÓD-ECG-12D' -> 'cod ecg 12d'."""
    texto = remover_acentos(texto.lower())
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto):
    """Limpa o texto, separa em palavras e remove as stopwords."""
    return [t for t in limpar_texto(texto).split() if t not in STOPWORDS_PT]


def texto_do_documento(doc):
    # indexamos titulo + conteudo juntos
    return doc["titulo"] + ". " + doc["conteudo"]


# =====================================================================
# FASE 2 - Motor lexico (BM25 Okapi)
# =====================================================================

def busca_bm25(consulta, k1, b):
    """Reconstroi o indice com os k1 e b atuais e devolve um score por documento."""
    docs_tokenizados = [tokenizar(texto_do_documento(d)) for d in CORPUS]
    bm25 = BM25Okapi(docs_tokenizados, k1=k1, b=b)
    scores = bm25.get_scores(tokenizar(consulta))
    # com k1=0 a lib pode gerar 0/0 (NaN) nos docs sem o termo, entao troca por 0
    return np.nan_to_num(np.array(scores, dtype=float))


# =====================================================================
# FASE 3 - Motor semantico vetorial
# =====================================================================
# Opcao A (preferida): embeddings reais com sentence-transformers.
# Opcao B (fallback): SIMULACAO VETORIAL. Cada dimensao do vetor e um "conceito
#   medico" e o valor e quantas palavras-gatilho daquele conceito aparecem no texto.
#   Assim "infarto", "ataque cardiaco" e "sindrome coronariana" caem na mesma
#   dimensao, o que imita o comportamento de um embedding (sinonimos ficam perto).

CONCEITOS = {
    "infarto":         ["infarto", "ataque cardiaco", "miocardio", "iam"],
    "coronariana/SCA": ["infarto", "ataque cardiaco", "coronari", "isquemia miocardica",
                        "precordial", "dor no peito", "angina"],
    "exame ECG":       ["ecg", "eletrocardiograma", "12d", "exame cardiaco", "telemetria",
                        "monitorizacao cardiaca"],
    "antiagregante":   ["aas", "acido acetilsalicilico", "aspirina", "antiagregante",
                        "plaquetario", "remedio para o coracao"],
    "hipertensao":     ["hipertens", "pressao arterial", "pressao alta"],
    "AVC":             ["avc", "acidente vascular", "derrame", "cerebral"],
    "trombolise":      ["trombolitico", "trombolise", "alteplase"],
    "parada cardiaca": ["parada cardio", "parada cardiaca", "pcr", "rcr", "reanimacao", "desfibrilacao",
                        "compressoes toracicas", "codigo azul"],
    "UTI/monitorar":   ["uti", "terapia intensiva", "monitor"],
    "coracao (geral)": ["cardi", "coracao"],
    "urgencia/tempo":  ["mortalidade", "minutos", "horas", "urgencia", "emergencia", "imediato"],
    "arritmia":        ["arritmia", "fibrilacao", "taquicardia", "palpitacao"],
}


def vetor_simulado(texto):
    limpo = " " + limpar_texto(texto) + " "
    vetor = np.zeros(len(CONCEITOS))
    for i, palavras in enumerate(CONCEITOS.values()):
        vetor[i] = sum(1 for p in palavras if p in limpo)
    return vetor


@st.cache_resource
def carregar_modelo_embeddings():
    """Tenta carregar o modelo de verdade. Se falhar, devolve None (usa simulacao)."""
    if not TEM_SENTENCE_TRANSFORMERS:
        return None
    try:
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


def gerar_embeddings(textos, modelo):
    if modelo is not None:
        return np.array(modelo.encode(textos))
    return np.array([vetor_simulado(t) for t in textos])


@st.cache_resource
def embeddings_do_corpus(_modelo):
    return gerar_embeddings([texto_do_documento(d) for d in CORPUS], _modelo)


def similaridade_cosseno(vetor_consulta, matriz_docs):
    """cos(q, d) = (q . d) / (|q| * |d|)  -- devolve 0 se algum vetor for nulo."""
    normas = np.linalg.norm(matriz_docs, axis=1) * np.linalg.norm(vetor_consulta)
    normas[normas == 0] = 1  # evita divisao por zero (o produto interno tambem sera 0)
    return matriz_docs @ vetor_consulta / normas


def busca_semantica(consulta, modelo):
    matriz_docs = embeddings_do_corpus(modelo)
    vetor_consulta = gerar_embeddings([consulta], modelo)[0]
    return similaridade_cosseno(vetor_consulta, matriz_docs)


# =====================================================================
# FASE 4 - Fusao RRF (Reciprocal Rank Fusion)
# =====================================================================

K_RRF = 60


def calcular_ranks(scores, so_positivos=False):
    """Posicao de cada doc (1 = melhor). Empate: fica na frente quem vem primeiro no corpus.
    Com so_positivos=True, doc com score <= 0 NAO foi recuperado pelo motor e fica sem
    rank (NaN) -- senao o desempate por ordem do corpus daria pontos de RRF de graca."""
    scores = np.array(scores, dtype=float)
    ranks = pd.Series(scores).rank(ascending=False, method="first").to_numpy(dtype=float, copy=True)
    if so_positivos:
        ranks[scores <= 0] = np.nan
    return ranks


def fusao_rrf(rank_bm25, rank_sem, alpha, k=K_RRF):
    """Score_RRF(D) = alpha/(k + rank_bm25) + (1 - alpha)/(k + rank_sem)
    Doc sem rank (nao recuperado) contribui com 0 naquele motor."""
    contrib_bm25 = alpha * np.where(np.isnan(rank_bm25), 0.0, 1.0 / (k + rank_bm25))
    contrib_sem = (1 - alpha) * np.where(np.isnan(rank_sem), 0.0, 1.0 / (k + rank_sem))
    return contrib_bm25, contrib_sem, contrib_bm25 + contrib_sem


def executar_busca(consulta, k1, b, alpha, modelo):
    """Roda as 3 buscas e junta tudo num DataFrame (1 linha por documento)."""
    scores_bm25 = busca_bm25(consulta, k1, b)
    scores_sem = busca_semantica(consulta, modelo)
    rank_bm25 = calcular_ranks(scores_bm25, so_positivos=True)
    rank_sem = calcular_ranks(scores_sem, so_positivos=True)
    c_bm25, c_sem, score_rrf = fusao_rrf(rank_bm25, rank_sem, alpha)

    return pd.DataFrame({
        "ID": [d["id"] for d in CORPUS],
        "Título": [d["titulo"] for d in CORPUS],
        "Trecho": [d["conteudo"] for d in CORPUS],
        "Score BM25": scores_bm25,
        "Rank BM25": pd.array(rank_bm25, dtype="Int64"),
        "Score Semântico": scores_sem,
        "Rank Semântico": pd.array(rank_sem, dtype="Int64"),
        "Contrib. BM25": c_bm25,
        "Contrib. Semântica": c_sem,
        "Score RRF": score_rrf,
        "Rank RRF": calcular_ranks(score_rrf).astype(int),
    })


# =====================================================================
# BONUS - Re-ranking com Cross-Encoder (Top-3 do RRF)
# =====================================================================

@st.cache_resource
def carregar_cross_encoder():
    if not TEM_SENTENCE_TRANSFORMERS:
        return None
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


def rerank_cross_encoder(consulta, df, cross_encoder):
    top3 = df.sort_values("Rank RRF").head(3).copy()
    pares = [(consulta, texto_do_documento(CORPUS[i])) for i in top3.index]
    top3["Nota Cross-Encoder"] = cross_encoder.predict(pares)
    top3["Nova posição"] = top3["Nota Cross-Encoder"].rank(ascending=False, method="first").astype(int)
    top3["Variação"] = top3["Rank RRF"] - top3["Nova posição"]  # positivo = subiu
    return top3.sort_values("Nova posição")


# =====================================================================
# Interface Streamlit
# =====================================================================

def mostrar_tabela(df, coluna_score, coluna_rank):
    """Tabela ordenada pelo rank escolhido."""
    tabela = df.sort_values(coluna_score, ascending=False, kind="stable")
    tabela = tabela[[coluna_rank, "ID", "Título", coluna_score, "Trecho"]]
    st.dataframe(tabela, hide_index=True, use_container_width=True)


def main():
    st.set_page_config(page_title="HealthSearch", page_icon="🩺", layout="wide")
    st.title("🩺 HealthSearch — Busca Híbrida (BM25 + Semântica + RRF)")
    st.caption("HealthTech Solutions · UNIPÊ · Recuperação de Informação")

    modelo = carregar_modelo_embeddings()

    # ---------- Sidebar: parametros ----------
    st.sidebar.header("⚙️ Calibração")
    k1 = st.sidebar.slider("k1 (saturação de frequência)", 0.0, 3.0, 1.2, 0.1)
    b = st.sidebar.slider("b (normalização por tamanho)", 0.0, 1.0, 0.75, 0.05)
    alpha = st.sidebar.slider("α (peso BM25 no RRF)", 0.0, 1.0, 0.5, 0.05,
                              help="α=1 usa só o BM25; α=0 usa só a busca semântica.")
    st.sidebar.caption(f"k_RRF = {K_RRF} (fixo)")
    usar_rerank = st.sidebar.checkbox("Re-ranking Cross-Encoder no Top-3 (bônus)")

    if modelo is None:
        st.sidebar.warning("Modo SIMULAÇÃO vetorial (sentence-transformers indisponível). "
                           "Os vetores são contagens de conceitos médicos.")
    else:
        st.sidebar.success("Embeddings reais: paraphrase-multilingual-MiniLM-L12-v2")

    # ---------- Consulta ----------
    exemplos = ["(digitar minha consulta)", "infarto", "ataque cardíaco", "CÓD-ECG-12D", "AAS 100mg",
                "AVC", "parada cardíaca"]
    col1, col2 = st.columns([3, 1])
    with col2:
        exemplo = st.selectbox("Consultas de teste", exemplos)
    with col1:
        padrao = "ataque cardíaco" if exemplo == exemplos[0] else exemplo
        consulta = st.text_input("Consulta clínica", value=padrao, key=f"q_{exemplo}")

    if not consulta.strip():
        st.info("Digite uma consulta para começar.")
        return

    inicio = time.perf_counter()
    df = executar_busca(consulta, k1, b, alpha, modelo)
    tempo_ms = (time.perf_counter() - inicio) * 1000

    # ---------- Metricas ----------
    top_bm25 = df.loc[df["Score BM25"].idxmax(), "ID"] if df["Score BM25"].max() > 0 else "—"
    top_sem = df.loc[df["Score Semântico"].idxmax(), "ID"] if df["Score Semântico"].max() > 0 else "—"
    top_rrf = df.loc[df["Score RRF"].idxmax(), "ID"] if df["Score RRF"].max() > 0 else "—"
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Top-1 BM25", top_bm25)
    m2.metric("Top-1 Semântico", top_sem)
    m3.metric("Top-1 Híbrido", top_rrf)
    m4.metric("Docs com match léxico", int((df["Score BM25"] > 0).sum()))
    m5.metric("Tempo da busca", f"{tempo_ms:.1f} ms")

    # ---------- Abas ----------
    aba_lex, aba_sem, aba_rrf, aba_matriz = st.tabs(
        ["🔤 Léxico (BM25)", "🧠 Semântico", "🔀 Híbrido RRF", "📊 Matriz Comparativa"])

    with aba_lex:
        st.subheader("Ranking BM25 Okapi")
        st.write(f"Tokens da consulta (após limpeza): `{tokenizar(consulta)}`")
        if df["Score BM25"].max() == 0:
            st.warning("Nenhum documento tem os termos da consulta (BM25 = 0 em tudo). "
                       "É o ponto cego do método léxico com sinônimos.")
        mostrar_tabela(df, "Score BM25", "Rank BM25")

    with aba_sem:
        st.subheader("Ranking por similaridade de cosseno")
        if df["Score Semântico"].max() == 0:
            st.warning("A consulta não ficou próxima de nenhum documento no espaço vetorial.")
        mostrar_tabela(df, "Score Semântico", "Rank Semântico")

    with aba_rrf:
        st.subheader("Fusão RRF")
        st.latex(r"Score_{RRF}(D) = \alpha \cdot \frac{1}{k + Rank_{BM25}} + "
                 r"(1-\alpha) \cdot \frac{1}{k + Rank_{Sem}}, \quad k = 60")
        mostrar_tabela(df, "Score RRF", "Rank RRF")

        if usar_rerank:
            st.markdown("#### Re-ranking com Cross-Encoder (Top-3)")
            cross_encoder = carregar_cross_encoder()
            if cross_encoder is None:
                st.error("Não consegui carregar o cross-encoder/ms-marco-MiniLM-L-6-v2 "
                         "(precisa de sentence-transformers e internet na 1ª execução).")
            else:
                rerank = rerank_cross_encoder(consulta, df, cross_encoder)
                st.dataframe(rerank[["ID", "Título", "Rank RRF", "Score RRF",
                                     "Nota Cross-Encoder", "Nova posição", "Variação"]],
                             hide_index=True, use_container_width=True)
                st.caption("Variação positiva = o documento subiu de posição depois do re-ranking. "
                           "Esse modelo foi treinado em inglês, então em português a nota é só indicativa.")

    with aba_matriz:
        st.subheader("Matriz comparativa de ranks")
        matriz = df[["ID", "Título", "Rank BM25", "Rank Semântico", "Rank RRF"]].copy()
        matriz["Δ RRF vs BM25"] = matriz["Rank BM25"] - matriz["Rank RRF"]
        matriz["Δ RRF vs Semântico"] = matriz["Rank Semântico"] - matriz["Rank RRF"]
        st.dataframe(matriz.sort_values("Rank RRF"), hide_index=True, use_container_width=True)
        st.caption("Δ positivo = o RRF colocou o documento numa posição melhor que o método isolado. "
                   "Rank vazio = o motor não recuperou o documento (score 0) e ele não pontua no RRF.")

        st.markdown("**Contribuição de cada motor no score RRF**")
        grafico = df.set_index("ID")[["Contrib. BM25", "Contrib. Semântica"]]
        st.bar_chart(grafico)


if __name__ == "__main__":
    main()

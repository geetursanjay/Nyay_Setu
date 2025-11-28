# nyayasetu_rag_slm_simple.py
import streamlit as st
import pandas as pd
import numpy as np
from gtts import gTTS
from io import BytesIO
from datetime import datetime
import os
import re

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDINGS_AVAILABLE = False

try:
    from rapidfuzz import process, fuzz
    FUZZY_AVAILABLE = True
except Exception:
    FUZZY_AVAILABLE = False

# -------------------------------
# Page config & CSS
# -------------------------------
st.set_page_config(layout="wide", page_title="Nyayasetu - AI Legal Consultant (RAG)")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    .stApp { background-image: url("https://raw.githubusercontent.com/geetursanjay/Nyay_Setu/main/background.png"); background-size: cover; }
    .stApp::before { content: ""; position: absolute; top:0; left:0; width:100%; height:100%; background-color: rgba(0,0,0,0.45); z-index:-1; }
    .main-header { display:flex; align-items:center; justify-content:center; gap:12px; font-family:'Dancing Script', cursive; }
    .main-header h1 { margin:0; color:#0b3d91; }
    .panel { background: rgba(255,255,255,0.88); padding:16px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Cache-safe loader for .gz or .csv
# -------------------------------
@st.cache_data
def load_csv_candidates():
    candidates = ["train.csv.gz", "test.csv.gz", "train.csv", "test.csv"]
    df_list = []
    sources = []
    errors = []

    def try_read(path):
        encs = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_exc = None
        for enc in encs:
            try:
                df = pd.read_csv(path, encoding=enc, compression="infer")
                return df, None
            except Exception as e:
                last_exc = e
        return None, str(last_exc)

    for fname in candidates:
        if os.path.exists(fname):
            df_load, err = try_read(fname)
            if df_load is not None:
                df_load.columns = [str(c) for c in df_load.columns]
                df_load["_source"] = fname
                df_list.append(df_load)
                sources.append(fname)
            else:
                errors.append(f"{fname}: {err}")

    if not df_list:
        return pd.DataFrame(), [], errors

    try:
        df_combined = pd.concat(df_list, ignore_index=True, sort=False)
    except Exception as e:
        return pd.DataFrame(), sources, [f"Concatenation failed: {e}"]

    return df_combined.reset_index(drop=True), sources, errors

df, detected_sources, load_errors = load_csv_candidates()
if load_errors:
    for e in load_errors:
        st.warning(f"⚠️ {e}")

if detected_sources:
    st.sidebar.info(f"Loaded: {', '.join(detected_sources)} — rows: {len(df)}")
else:
    st.sidebar.info("No dataset files found; place train.csv.gz/test.csv.gz in app folder.")

# -------------------------------
# Column detection (simple, no manual mapping UI)
# -------------------------------
def normalize(c): 
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip()).lower()

cols = df.columns.tolist() if not df.empty else []
norm_map = { normalize(c): c for c in cols }

# 1) Try to detect language-specific Query_<Language> columns
LANGS = ["English","Hindi","Bengali","Tamil","Telugu","Marathi"]
available_languages = []
lang_columns = {}  # e.g. "English": {"query": "Query_English", "short": ..., "detailed": ...}

for lang in LANGS:
    key = lang.lower()
    qname = next((norm_map[n] for n in norm_map if n == f"query_{key}" or n == f"query{key}"), None)
    sname = next((norm_map[n] for n in norm_map if n == f"short_{key}" or n == f"short{key}"), None)
    dname = next((norm_map[n] for n in norm_map if n.startswith(f"detailed") and key in n or n.startswith(f"detail") and key in n), None)
    # fallback contains
    if not qname:
        qname = next((norm_map[n] for n in norm_map if n.startswith("query") and key in n), None)
    if not sname:
        sname = next((norm_map[n] for n in norm_map if n.startswith("short") and key in n), None)
    if qname or sname or dname:
        available_languages.append(lang)
        lang_columns[lang] = {"query": qname, "short": sname, "detailed": dname}

# 2) If no language-specific columns found, fallback to generic query/short/detailed detection
generic_query_col = None
generic_short_col = None
generic_detailed_col = None

if not available_languages:
    # prefer columns named query, question, q_, prompt
    for pattern in ["query", "question", "q_", "q"]:
        generic_query_col = generic_query_col or next((norm_map[n] for n in norm_map if pattern in n), None)
    for pattern in ["short", "summary", "ans", "answer"]:
        generic_short_col = generic_short_col or next((norm_map[n] for n in norm_map if pattern in n), None)
    for pattern in ["detailed", "detail", "explain"]:
        generic_detailed_col = generic_detailed_col or next((norm_map[n] for n in norm_map if pattern in n), None)

# -------------------------------
# Sidebar: minimal (only language selector)
# -------------------------------
st.sidebar.header("Settings")
if available_languages:
    selected_lang = st.sidebar.selectbox("Select language:", available_languages)
    col_query = lang_columns[selected_lang].get("query") or lang_columns[selected_lang].get("query")  # explicit
    col_short = lang_columns[selected_lang].get("short")
    col_detailed = lang_columns[selected_lang].get("detailed")
else:
    st.sidebar.warning("No Query_<Language> columns found. Using generic detection (no language labels).")
    selected_lang = "Default"
    col_query = generic_query_col
    col_short = generic_short_col
    col_detailed = generic_detailed_col

# show detected columns (compact)
st.sidebar.markdown("---")
st.sidebar.markdown("**Detected columns**")
st.sidebar.write({
    "Query": col_query,
    "Short": col_short,
    "Detailed": col_detailed
})

# Continue even if some columns are missing — show a visible warning but do not block
if col_query is None:
    st.warning("No query/question column detected. Upload a dataset with Query_<Language> or a generic question column.")
    # do not stop() — user can still enter question (it will yield no match)

# -------------------------------
# RAG: model + embeddings (optional)
# -------------------------------
@st.cache_resource
def load_model():
    if EMBEDDINGS_AVAILABLE:
        try:
            return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            return None
    return None

embedding_model = load_model()

@st.cache_data
def build_corpus(query_col):
    if query_col is None or query_col not in df.columns:
        return [], None
    queries = df[query_col].dropna().astype(str).tolist()
    if embedding_model is None or not queries:
        return queries, None
    try:
        emb = embedding_model.encode(queries, show_progress_bar=False)
        return queries, emb
    except Exception:
        return queries, None

queries_list, corpus_embeddings = build_corpus(col_query)

# -------------------------------
# UI Header and examples
# -------------------------------
st.markdown("<div class='main-header'><span>⚖️</span><h1>Nyayasetu - AI Legal Consultant</h1></div>", unsafe_allow_html=True)
st.markdown("Nyayasetu provides quick multi-language legal guidance using RAG when available.")
st.markdown("---")

# example queries (if available)
if col_query and col_query in df.columns:
    example_queries = df[col_query].dropna().astype(str).tolist()[:3]
else:
    example_queries = []

st.markdown("### Get Legal Guidance Instantly")
st.markdown("**Try one of these example questions:**")

if example_queries:
    cols = st.columns(len(example_queries))
    for i, ex in enumerate(example_queries):
        with cols[i]:
            if st.button(ex if len(ex) < 60 else ex[:57] + "...", key=f"ex_{i}"):
                st.session_state["user_q"] = ex

# -------------------------------
# Input & search
# -------------------------------
user_q = st.text_input("Enter your legal question:", value=st.session_state.get("user_q",""), key="input_q")
use_rag = st.checkbox("Use RAG semantic search (recommended)", value=(embedding_model is not None and corpus_embeddings is not None), disabled=(embedding_model is None or corpus_embeddings is None))

if st.button("Get Answer"):
    if not user_q or user_q.strip()=="":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            matched_row = None
            confidence = 0.0

            # RAG path (if available & enabled)
            if use_rag and corpus_embeddings is not None:
                try:
                    q_emb = embedding_model.encode([user_q])
                    scores = np.dot(corpus_embeddings, q_emb.T).flatten()
                    top_idx = int(np.argmax(scores))
                    top_score = float(scores[top_idx])
                    # threshold is empirical; tune if needed
                    if top_score > 0.25:
                        matched_row = df.iloc[top_idx]
                        confidence = float(top_score)*100.0
                        st.success(f"RAG matched (score {top_score:.3f})")
                except Exception:
                    matched_row = None

            # Fuzzy fallback
            if matched_row is None and FUZZY_AVAILABLE and queries_list:
                try:
                    best = process.extractOne(user_q, queries_list, scorer=fuzz.WRatio)
                    if best:
                        match_text, score, pos = best
                        if score >= 50:
                            idxs = df[df[col_query].astype(str)==match_text].index
                            if len(idxs)>0:
                                matched_row = df.loc[idxs[0]]
                                confidence = float(score)
                                st.success(f"Fuzzy matched (sim {score:.1f}%)")
                except Exception:


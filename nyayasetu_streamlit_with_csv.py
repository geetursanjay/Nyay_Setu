# nyayasetu_clean_slm_rag.py
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
# Page config & simple CSS
# -------------------------------
st.set_page_config(layout="wide", page_title="Nyayasetu - AI Legal Consultant (RAG)")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    .stApp { background-image: url("https://raw.githubusercontent.com/geetursanjay/Nyay_Setu/main/background.png"); background-size: cover; background-repeat: no-repeat; }
    .stApp::before { content: ""; position: absolute; top:0; left:0; width:100%; height:100%; background-color: rgba(0,0,0,0.45); z-index:-1; }
    .main-header { display:flex; align-items:center; justify-content:center; gap:12px; font-family:'Dancing Script', cursive; }
    .main-header h1 { margin:0; color:#0b3d91; }
    .panel { background: rgba(255,255,255,0.9); padding:18px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Cache-safe loader for CSV / CSV.GZ
# -------------------------------
@st.cache_data
def load_dataset_files():
    """
    Load train/test CSV or compressed CSV (.gz).
    Returns: (df_combined, sources_list, errors_list)
    """
    candidates = ["train.csv.gz", "test.csv.gz", "train.csv", "test.csv"]
    df_parts = []
    sources = []
    errors = []

    def try_read(path):
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_exc = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, compression="infer")
                return df, None
            except Exception as e:
                last_exc = e
        return None, str(last_exc)

    for fname in candidates:
        if os.path.exists(fname):
            df_loaded, err = try_read(fname)
            if df_loaded is not None:
                df_loaded.columns = [str(c) for c in df_loaded.columns]
                df_loaded["_source"] = fname
                df_parts.append(df_loaded)
                sources.append(fname)
            else:
                errors.append(f"{fname}: {err}")

    if not df_parts:
        return pd.DataFrame(), [], errors

    try:
        combined = pd.concat(df_parts, ignore_index=True, sort=False)
    except Exception as e:
        errors.append(f"Concatenation failed: {e}")
        return pd.DataFrame(), sources, errors

    return combined.reset_index(drop=True), sources, errors

df, detected_sources, load_errors = load_dataset_files()

# show loader errors (UI outside cached)
if load_errors:
    for e in load_errors:
        st.warning(f"⚠️ {e}")

if not detected_sources:
    st.error("No dataset files found. Please put train.csv.gz/test.csv.gz (or train.csv/test.csv) in the app folder.")
    st.stop()

# -------------------------------
# Simple column detection (no manual mapping UI)
# -------------------------------
def normalize(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s)
    return s.lower()

cols = df.columns.tolist() if not df.empty else []
norm_map = { normalize(c): c for c in cols }

LANGS = ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi"]
available_languages = []
lang_cols = {}

for lang in LANGS:
    key = lang.lower()
    q_col = next((norm_map[n] for n in norm_map if n == f"query_{key}" or n == f"query{key}"), None)
    # fallback patterns
    if q_col is None:
        q_col = next((norm_map[n] for n in norm_map if n.startswith("query") and key in n), None)
        q_col = q_col or next((norm_map[n] for n in norm_map if ("question" in n or n.startswith("q")) and key in n), None)
    s_col = next((norm_map[n] for n in norm_map if n == f"short_{key}" or n == f"short{key}"), None)
    if s_col is None:
        s_col = next((norm_map[n] for n in norm_map if n.startswith("short") and key in n), None)
    d_col = next((norm_map[n] for n in norm_map if (n.startswith("detailed") or n.startswith("detail")) and key in n), None)

    if q_col or s_col or d_col:
        available_languages.append(lang)
        lang_cols[lang] = {"query": q_col, "short": s_col, "detailed": d_col}

# If no language-specific columns, try generic detection
generic_query = None
generic_short = None
generic_detailed = None
if not available_languages:
    # common names to look for
    for patt in ["query", "question", "q_","q"]:
        generic_query = generic_query or next((norm_map[n] for n in norm_map if patt in n), None)
    for patt in ["short", "summary", "ans", "answer"]:
        generic_short = generic_short or next((norm_map[n] for n in norm_map if patt in n), None)
    for patt in ["detailed", "detail", "explain"]:
        generic_detailed = generic_detailed or next((norm_map[n] for n in norm_map if patt in n), None)

# -------------------------------
# Sidebar: only language selector (simplified UI)
# -------------------------------
st.sidebar.header("Settings")
if available_languages:
    selected_language = st.sidebar.selectbox("Select language:", available_languages)
    col_query = lang_cols[selected_language].get("query")
    col_short = lang_cols[selected_language].get("short")
    col_detailed = lang_cols[selected_language].get("detailed")
else:
    st.sidebar.warning("No Query_<Language> columns detected. Using generic detection.")
    selected_language = "Default"
    col_query = generic_query
    col_short = generic_short
    col_detailed = generic_detailed

# compact display of detection
st.sidebar.markdown("---")
st.sidebar.markdown("Detected columns:")
st.sidebar.write({"query": col_query, "short": col_short, "detailed": col_detailed})

# do not block UI if query column missing; show warning
if col_query is None:
    st.warning("No query/question column detected. The app will run but won't find matches until a query column is present.")
# -------------------------------
# RAG: model & corpus preparation
# -------------------------------
@st.cache_resource
def get_embedding_model():
    if EMBEDDINGS_AVAILABLE:
        try:
            return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            # return None if model load fails
            return None
    return None

embedding_model = get_embedding_model()

@st.cache_data
def build_queries_and_embeddings(query_col_name):
    if query_col_name is None or query_col_name not in df.columns:
        return [], None
    queries = df[query_col_name].dropna().astype(str).tolist()
    if not queries or embedding_model is None:
        return queries, None
    try:
        emb = embedding_model.encode(queries, show_progress_bar=False)
        return queries, emb
    except Exception:
        return queries, None

queries_list, corpus_embeddings = build_queries_and_embeddings(col_query)

# -------------------------------
# Header & examples UI
# -------------------------------
st.markdown("<div class='main-header'><span>⚖️</span><h1>Nyayasetu - AI Legal Consultant</h1></div>", unsafe_allow_html=True)
st.markdown("Nyayasetu provides quick multi-language legal guidance using RAG (if available).")
st.markdown("---")

# show dataset info
st.info(f"Loaded: {', '.join(detected_sources)} — entries: {len(df)}")
st.caption(f"Language: {selected_language} — Query column: `{col_query}`")

# example queries
example_queries = []
if col_query and col_query in df.columns:
    example_queries = df[col_query].dropna().astype(str).tolist()[:3]

st.markdown("### Get Legal Guidance Instantly")
st.markdown("**Try one of these example questions:**")
if example_queries:
    ex_cols = st.columns(len(example_queries))
    for i, ex in enumerate(example_queries):
        with ex_cols[i]:
            if st.button(ex if len(ex) <= 60 else ex[:57] + "...", key=f"example_{i}"):
                st.session_state['user_q'] = ex

# -------------------------------
# Input & Retrieval
# -------------------------------
user_question = st.text_input("Enter your legal question:", value=st.session_state.get("user_q", ""), key="user_input")
use_rag = st.checkbox("Use RAG semantic search (recommended)", value=(embedding_model is not None and corpus_embeddings is not None), disabled=(embedding_model is None or corpus_embeddings is None))

if st.button("Get Answer"):
    if not user_question or user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            matched_row = None
            confidence = 0.0

            # RAG path
            if use_rag and corpus_embeddings is not None and embedding_model is not None:
                try:
                    q_emb = embedding_model.encode([user_question])
                    scores = np.dot(corpus_embeddings, q_emb.T).flatten()
                    top_idx = int(np.argmax(scores))
                    top_score = float(scores[top_idx])
                    # threshold (tunable)
                    if top_score > 0.25:
                        matched_row = df.iloc[top_idx]
                        confidence = float(top_score) * 100.0
                        st.success(f"RAG match found (score {top_score:.3f})")
                except Exception as e:
                    st.warning(f"RAG search failed: {e}")

            # Fuzzy fallback
            if matched_row is None and FUZZY_AVAILABLE and queries_list:
                try:
                    best = process.extractOne(user_question, queries_list, scorer=fuzz.WRatio)
                    if best:
                        match_text, score, pos = best
                        if score >= 50:
                            idxs = df[df[col_query].astype(str) == match_text].index
                            if not idxs.empty:
                                matched_row = df.loc[idxs[0]]
                                confidence = float(score)
                                st.success(f"Fuzzy match found (similarity {score:.1f}%)")
                except Exception as e:
                    st.warning(f"Fuzzy search failed: {e}")

            # Prepare answers
            if matched_row is not None:
                short_answer = matched_row.get(col_short, "Short answer not available") if col_short else "Short answer not available"
                detailed_answer = matched_row.get(col_detailed, short_answer) if col_detailed else short_answer
            else:
                short_answer = "❌ No relevant answer found. Try rephrasing or consult a legal expert."
                detailed_answer = short_answer
                st.warning("No sufficiently relevant match found.")

        # Display results
        st.markdown("---")
        st.markdown("#### Short Answer")
        st.info(str(short_answer))
        st.markdown("#### Detailed Answer")
        with st.expander("View full details", expanded=False):
            st.write(str(detailed_answer))

        # Text-to-Speech playback
        try:
            lang_code_map = {"English": "en", "Hindi": "hi", "Bengali": "bn", "Tamil": "ta", "Telugu": "te", "Marathi": "mr"}
            lang_code = lang_code_map.get(selected_language, "en")
            tts = gTTS(text=str(short_answer), lang=lang_code, slow=False)
            buf = BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf.read(), format="audio/mp3")
        except Exception as e:
            st.warning(f"Audio generation failed: {e}")

        # Save into session history
        st.session_state.setdefault("chat_history", [])
        st.session_state["chat_history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": user_question,
            "answer": str(short_answer),
            "language": selected_language,
            "confidence": float(confidence)
        })

# -------------------------------
# Recent history
# -------------------------------
if st.session_state.get("chat_history"):
    st.markdown("---")
    st.markdown("### Recent queries")
    for entry in reversed(st.session_state["chat_history"][-8:]):
        st.write(f"Q: {entry['question']}")
        st.write(f"A: {entry['answer']}")
        st.caption(f"{entry['timestamp']} | {entry['language']} | conf: {entry.get('confidence',0):.1f}")
        st.markdown("---")

st.markdown("---")
st.caption("⚠️ Disclaimer: This assistant provides general legal information only and is not a substitute for professional legal advice.")

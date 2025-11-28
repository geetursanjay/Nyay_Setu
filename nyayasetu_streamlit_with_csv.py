# nyayasetu_final.py
# Single-page Streamlit app (SLM) using your dataset's Title/Summary/Case columns.
# - Loads train.csv.gz / test.csv.gz (or plain .csv)
# - Treats Title as the query/corpus, Summary as short answer, Case as detailed answer
# - Uses RAG (sentence-transformers) if available, otherwise falls back to rapidfuzz fuzzy matching
# - Plays short answers using gTTS
# Paste into a file and run: streamlit run nyayasetu_final.py

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
st.set_page_config(layout="wide", page_title="Nyayasetu — Legal Assistant (Title→Summary→Case)")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    .stApp { background-image: url("https://raw.githubusercontent.com/geetursanjay/Nyay_Setu/main/background.png"); background-size: cover; background-repeat: no-repeat; }
    .stApp::before { content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.45); z-index:-1; }
    .main-header { display:flex; align-items:center; justify-content:center; gap:12px; font-family:'Dancing Script', cursive; }
    .panel { background: rgba(255,255,255,0.92); padding:16px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Cache-safe dataset loader
# -------------------------------
@st.cache_data
def load_csvs():
    """
    Try to load train.csv.gz / test.csv.gz / train.csv / test.csv (in that order).
    Returns: (df_combined, sources, errors)
    """
    candidates = ["train.csv.gz", "test.csv.gz", "train.csv", "test.csv"]
    parts = []
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
            df_load, err = try_read(fname)
            if df_load is not None:
                df_load.columns = [str(c) for c in df_load.columns]
                df_load["_source"] = fname
                parts.append(df_load)
                sources.append(fname)
            else:
                errors.append(f"{fname}: {err}")

    if not parts:
        return pd.DataFrame(), [], errors

    try:
        combined = pd.concat(parts, ignore_index=True, sort=False)
    except Exception as e:
        return pd.DataFrame(), sources, [f"Concatenation failed: {e}"]

    return combined.reset_index(drop=True), sources, errors

df, detected_sources, load_errors = load_csvs()

# Show loader errors (outside cached function)
if load_errors:
    for e in load_errors:
        st.warning(f"⚠️ {e}")

if not detected_sources:
    st.error("No dataset files found. Please place train.csv.gz and/or test.csv.gz (or train.csv/test.csv) in the app folder.")
    st.stop()

# -------------------------------
# Map dataset columns (Title/ Summary / Case) to Query/Short/Detailed
# -------------------------------
# Prefer Title -> Query, Summary -> Short, Case -> Detailed.
# If Title missing, try other likely columns; if Summary missing, fallback to Title as short.
col_query = None
col_short = None
col_detailed = None

cols = df.columns.tolist()

if "Title" in cols:
    col_query = "Title"
elif "title" in [c.lower() for c in cols]:
    # find case-insensitive
    col_query = next(c for c in cols if c.lower() == "title")

# Short / summary
if "Summary" in cols:
    col_short = "Summary"
elif "summary" in [c.lower() for c in cols]:
    col_short = next(c for c in cols if c.lower() == "summary")

# Detailed / Case
if "Case" in cols:
    col_detailed = "Case"
elif "case" in [c.lower() for c in cols]:
    col_detailed = next(c for c in cols if c.lower() == "case")

# Fallbacks
if col_query is None:
    # try to auto-detect a question-like column (heuristic)
    candidate = None
    for pattern in ["question", "query", "prompt", "issue", "title", "subject", "text"]:
        candidate = candidate or next((c for c in cols if pattern in c.lower()), None)
    col_query = candidate

if col_short is None:
    # try to choose a concise text column
    candidate = None
    for pattern in ["summary", "short", "answer", "abstract"]:
        candidate = candidate or next((c for c in cols if pattern in c.lower()), None)
    col_short = candidate or col_query  # if nothing else, use query as short

if col_detailed is None:
    candidate = None
    for pattern in ["case", "details", "detailed", "full", "text", "body"]:
        candidate = candidate or next((c for c in cols if pattern in c.lower()), None)
    col_detailed = candidate or col_short

# Force single-language behavior
available_language = "English"
selected_language = available_language

# Display mapping summary
st.markdown("<div class='main-header'><span>⚖️</span><h1>Nyayasetu — Legal Assistant</h1></div>", unsafe_allow_html=True)
st.markdown("Quick mapping: `Title` → Query, `Summary` → Short answer, `Case` → Detailed answer. (Single-language mode)")
st.markdown("---")
st.info(f"Loaded sources: {', '.join(detected_sources)} — total rows: {len(df)}")
st.write("**Column mapping detected:**")
st.write({"query": col_query, "short": col_short, "detailed": col_detailed})

# If no query column after heuristics, prompt user to pick one from actual columns
if col_query is None:
    st.warning("No query-like column auto-detected. Please select which column to use as the query/title.")
    chosen = st.selectbox("Select Query column from dataset", options=["-- none --"] + cols)
    if chosen != "-- none --":
        col_query = chosen
    else:
        st.error("App needs a Query column to match user questions. Upload/rename dataset or pick a column.")
        st.stop()

# Build queries list and embeddings (RAG)
@st.cache_resource
def load_embedding_model():
    if EMBEDDINGS_AVAILABLE:
        try:
            return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except Exception:
            return None
    return None

embedding_model = load_embedding_model()

@st.cache_data
def build_corpus(query_col):
    if query_col not in df.columns:
        return [], None
    queries = df[query_col].dropna().astype(str).tolist()
    if not queries or embedding_model is None:
        return queries, None
    try:
        emb = embedding_model.encode(queries, show_progress_bar=False)
        return queries, emb
    except Exception:
        return queries, None

queries_list, corpus_embeddings = build_corpus(col_query)

# -------------------------------
# Interface: input + search
# -------------------------------
st.markdown("### Ask a legal question (search over dataset Titles)")
user_q = st.text_input("Enter your question or keywords:", value="", placeholder="e.g., eviction without notice")
use_rag = st.checkbox("Use RAG semantic search (recommended)", value=(embedding_model is not None and corpus_embeddings is not None), disabled=(embedding_model is None or corpus_embeddings is None))

if st.button("Get Answer"):
    if not user_q or user_q.strip() == "":
        st.warning("Please enter a question or some keywords.")
    else:
        with st.spinner("Searching the dataset..."):
            matched_row = None
            confidence = 0.0

            # RAG path
            if use_rag and corpus_embeddings is not None and embedding_model is not None:
                try:
                    q_emb = embedding_model.encode([user_q])
                    scores = np.dot(corpus_embeddings, q_emb.T).flatten()
                    top_idx = int(np.argmax(scores))
                    top_score = float(scores[top_idx])
                    # threshold can be tuned; 0.25 is conservative
                    if top_score > 0.25:
                        matched_row = df.iloc[top_idx]
                        confidence = float(top_score) * 100.0
                        st.success(f"RAG matched (score={top_score:.3f})")
                except Exception as e:
                    st.warning(f"RAG search failed: {e}")

            # Fuzzy fallback
            if matched_row is None and FUZZY_AVAILABLE and queries_list:
                try:
                    best = process.extractOne(user_q, queries_list, scorer=fuzz.WRatio)
                    if best:
                        match_text, score, pos = best
                        if score >= 50:
                            idxs = df[df[col_query].astype(str) == match_text].index
                            if len(idxs) > 0:
                                matched_row = df.loc[idxs[0]]
                                confidence = float(score)
                                st.success(f"Fuzzy matched (similarity={score:.1f}%)")
                except Exception as e:
                    st.warning(f"Fuzzy search failed: {e}")

            # Prepare answers
            if matched_row is not None:
                short_ans = matched_row.get(col_short, "Short answer not available")
                detailed_ans = matched_row.get(col_detailed, short_ans)
            else:
                short_ans = "❌ No relevant answer found. Try rephrasing or consult a legal expert."
                detailed_ans = short_ans
                st.warning("No sufficiently relevant match found.")

        # Display results
        st.markdown("---")
        st.markdown("#### Short answer")
        st.info(str(short_ans))
        st.markdown("#### Detailed answer")
        with st.expander("View full details", expanded=False):
            st.write(str(detailed_ans))

        # Text-to-speech
        try:
            lang_code = "en"  # dataset single-language English
            tts = gTTS(text=str(short_ans), lang=lang_code, slow=False)
            buf = BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf.read(), format="audio/mp3")
        except Exception as e:
            st.warning(f"Audio generation failed: {e}")

        # Save chat history
        st.session_state.setdefault("chat_history", [])
        st.session_state["chat_history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": user_q,
            "answer": str(short_ans),
            "confidence": float(confidence)
        })

# -------------------------------
# Recent queries
# -------------------------------
if st.session_state.get("chat_history"):
    st.markdown("---")
    st.markdown("### Recent queries")
    for entry in reversed(st.session_state["chat_history"][-8:]):
        st.write(f"Q: {entry['question']}")
        st.write(f"A: {entry['answer']}")
        st.caption(f"{entry['timestamp']} | conf: {entry.get('confidence',0):.1f}")
        st.markdown("---")

st.markdown("---")
st.caption("⚠️ Disclaimer: This assistant provides general legal information only and is not a substitute for professional legal advice.")

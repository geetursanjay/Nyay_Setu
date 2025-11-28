
# nyayasetu_translated.py
import streamlit as st
import pandas as pd
import numpy as np
from gtts import gTTS
from io import BytesIO
from datetime import datetime
import os
import re

# Optional dependencies (RAG & fuzzy)
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

# Translation libs (try googletrans first, then deep_translator)
TRANSLATION_AVAILABLE = False
TRANSLATOR_TYPE = None
translator_google = None
GoogleTranslator = None

try:
    from googletrans import Translator as GoogleTransTranslator
    translator_google = GoogleTransTranslator()
    TRANSLATION_AVAILABLE = True
    TRANSLATOR_TYPE = "googletrans"
except Exception:
    # fallback to deep_translator
    try:
        from deep_translator import GoogleTranslator as DeepGoogleTranslator
        GoogleTranslator = DeepGoogleTranslator
        TRANSLATION_AVAILABLE = True
        TRANSLATOR_TYPE = "deep"
    except Exception:
        TRANSLATION_AVAILABLE = False
        TRANSLATOR_TYPE = None

# -------------------------------
# Page config & CSS
# -------------------------------
st.set_page_config(layout="wide", page_title="Nyayasetu — Legal Assistant (Translated)")
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
# Helper: load CSV / CSV.GZ (cache-safe)
# -------------------------------
@st.cache_data
def load_csv_candidates():
    candidates = ["train.csv.gz", "test.csv.gz", "train.csv", "test.csv"]
    df_list = []
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
                df_list.append(df_loaded)
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

# UI notifications for load errors
if load_errors:
    for e in load_errors:
        st.warning(f"⚠️ {e}")

if not detected_sources:
    st.error("No dataset files found. Please place train.csv.gz and/or test.csv.gz (or train.csv/test.csv) in the app folder.")
    st.stop()

# -------------------------------
# Map dataset columns: Title -> Query, Summary -> Short, Case -> Detailed
# -------------------------------
cols = df.columns.tolist()
col_query = None
col_short = None
col_detailed = None

# Prefer exact case names Title/Summary/Case, else case-insensitive
if "Title" in cols:
    col_query = "Title"
else:
    col_query = next((c for c in cols if c.lower() == "title"), None)

if "Summary" in cols:
    col_short = "Summary"
else:
    col_short = next((c for c in cols if c.lower() == "summary"), None)

if "Case" in cols:
    col_detailed = "Case"
else:
    col_detailed = next((c for c in cols if c.lower() == "case"), None)

# Fallback heuristics if any missing
if col_query is None:
    for patt in ["question", "query", "prompt", "issue", "title", "subject", "text"]:
        col_query = col_query or next((c for c in cols if patt in c.lower()), None)

if col_short is None:
    for patt in ["summary", "short", "answer", "abstract"]:
        col_short = col_short or next((c for c in cols if patt in c.lower()), None)
    col_short = col_short or col_query

if col_detailed is None:
    for patt in ["case", "details", "detailed", "full", "body", "text"]:
        col_detailed = col_detailed or next((c for c in cols if patt in c.lower()), None)
    col_detailed = col_detailed or col_short

# Single-language mode for dataset (since your dataset is Title/Summary/Case)
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr"
}

# Sidebar: always show language selector (UI requirement)
st.sidebar.header("Settings")
sidebar_languages = list(language_map.keys())
selected_language = st.sidebar.selectbox("Select language:", sidebar_languages, index=0)

# Info about translation availability
if TRANSLATION_AVAILABLE:
    st.sidebar.success(f"Translation available ({TRANSLATOR_TYPE})")
else:
    st.sidebar.info("Translation not available. Install googletrans or deep-translator to enable.")

# Show mapping summary
st.markdown("<div class='main-header'><span>⚖️</span><h1>Nyayasetu — Legal Assistant</h1></div>", unsafe_allow_html=True)
st.markdown("Mapping: `Title` → Query, `Summary` → Short answer, `Case` → Detailed answer (single-language dataset).")
st.markdown("---")
st.info(f"Loaded sources: {', '.join(detected_sources)} — total rows: {len(df)}")
st.write("**Detected column mapping:**")
st.json({"query": col_query, "short": col_short, "detailed": col_detailed})

if col_query is None:
    st.error("No query-like column detected. Please ensure your dataset has a Title/Question column.")
    st.stop()

# -------------------------------
# RAG: load model & build corpus embeddings (optional)
# -------------------------------
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
def build_queries_and_embeddings(query_col):
    if query_col not in df.columns:
        return [], None
    queries = df[query_col].dropna().astype(str).tolist()
    if embedding_model is None or not queries:
        return queries, None
    try:
        emb = embedding_model.encode(queries, show_progress_bar=False)
        return queries, emb
    except Exception:
        return queries, None

queries_list, corpus_embeddings = build_queries_and_embeddings(col_query)

# -------------------------------
# Helper: translate text (uses googletrans or deep-translator)
# -------------------------------
def translate_text(text: str, target_lang_code: str) -> str:
    if not TRANSLATION_AVAILABLE or target_lang_code == "en":
        return text
    try:
        if TRANSLATOR_TYPE == "googletrans" and translator_google is not None:
            # googletrans may auto-detect source
            res = translator_google.translate(text, dest=target_lang_code)
            return getattr(res, "text", str(res))
        elif TRANSLATOR_TYPE == "deep" and GoogleTranslator is not None:
            trans = GoogleTranslator(source="auto", target=target_lang_code)
            return trans.translate(text)
    except Exception:
        return text
    return text

# -------------------------------
# UI: examples, input, search
# -------------------------------
st.markdown("### Get Legal Guidance Instantly")
st.markdown("**Try one of these example questions:**")

example_queries = df[col_query].dropna().astype(str).tolist()[:3] if col_query in df.columns else []
if example_queries:
    ex_cols = st.columns(len(example_queries))
    for i, ex in enumerate(example_queries):
        with ex_cols[i]:
            if st.button(ex if len(ex) <= 60 else ex[:57] + "...", key=f"example_{i}"):
                st.session_state['user_q'] = ex

user_question = st.text_input("Enter your question or keywords:", value=st.session_state.get("user_q", ""), key="user_question_input")
use_rag = st.checkbox("Use RAG semantic search (recommended)", value=(embedding_model is not None and corpus_embeddings is not None), disabled=(embedding_model is None or corpus_embeddings is None))

if st.button("Get Answer"):
    if not user_question or user_question.strip() == "":
        st.warning("Please enter a question or keywords.")
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
                    if top_score > 0.25:
                        matched_row = df.iloc[top_idx]
                        confidence = float(top_score) * 100.0
                        st.success(f"RAG matched (score {top_score:.3f})")
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
                            if len(idxs) > 0:
                                matched_row = df.loc[idxs[0]]
                                confidence = float(score)
                                st.success(f"Fuzzy matched (similarity {score:.1f}%)")
                except Exception as e:
                    st.warning(f"Fuzzy search failed: {e}")

            if matched_row is not None:
                short_answer = str(matched_row.get(col_short, "Short answer not available")) if col_short else "Short answer not available"
                detailed_answer = str(matched_row.get(col_detailed, short_answer)) if col_detailed else short_answer
            else:
                short_answer = "❌ No relevant answer found. Try rephrasing or consult a legal expert."
                detailed_answer = short_answer
                st.warning("No sufficiently relevant match found.")

        # Translate short answer if needed
        target_code = language_map.get(selected_language, "en")
        translated_answer = translate_text(short_answer, target_code)

        # Display answers (show translated short answer)
        st.markdown("---")
        st.markdown("#### Short Answer")
        st.info(str(translated_answer))
        st.markdown("#### Detailed Answer")
        with st.expander("View full details", expanded=False):
            st.write(str(detailed_answer))

        # Text-to-Speech for translated answer
        try:
            # if translation failed or not available, tts will still attempt with target_code (may error)
            tts_lang = target_code
            tts = gTTS(text=str(translated_answer), lang=tts_lang, slow=False)
            buf = BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf.read(), format="audio/mp3")
        except Exception as e:
            st.warning(f"Audio generation failed: {e}")

        # Save to chat history
        st.session_state.setdefault("chat_history", [])
        st.session_state["chat_history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": user_question,
            "answer": str(translated_answer),
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


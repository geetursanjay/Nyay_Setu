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
st.set_page_config(layout="wide", page_title="Nyayasetu - AI Legal Assistant (RAG)")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    .stApp {
        background-image: url("https://raw.githubusercontent.com/geetursanjay/Nyay_Setu/main/background.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center center;
    }
    .stApp::before { content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                     background-color: rgba(0,0,0,0.45); z-index:-1; }
    .main-header { display:flex; align-items:center; justify-content:center; gap:12px; font-family:'Dancing Script', cursive; }
    .main-header h1{ margin:0; color:#0b3d91; }
    .panel { background: rgba(255,255,255,0.85); padding:16px; border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Utility: load CSV or CSV.GZ (cache-safe)
# -------------------------------
@st.cache_data
def load_compressed_data():
    """Load train/test CSV (accept .gz). Return (df_combined, sources, errors)."""
    candidates = ["train.csv.gz", "test.csv.gz", "train.csv", "test.csv"]
    df_list = []
    sources = []
    errors = []

    def try_read(path):
        encs = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_exc = None
        for enc in encs:
            try:
                # compression='infer' lets pandas open gz automatically
                df = pd.read_csv(path, encoding=enc, compression="infer")
                return df, None
            except Exception as e:
                last_exc = e
        return None, str(last_exc)

    for fname in candidates:
        if os.path.exists(fname):
            df_load, err = try_read(fname)
            if df_load is not None:
                # ensure string column names
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

# Load dataset and show errors/notes
df, detected_sources, load_errors = load_compressed_data()
if load_errors:
    for e in load_errors:
        st.warning(f"⚠️ {e}")

if not detected_sources:
    st.error("No dataset files found. Please place `train.csv.gz` and/or `test.csv.gz` in the app folder.")
    st.stop()

# -------------------------------
# Flexible column detection & mapping
# -------------------------------
def normalize_colname(c):
    if c is None:
        return ""
    s = str(c).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.lower()

# Build normalized map normalized -> original
normalized_map = { normalize_colname(col): col for col in df.columns.tolist() }

# Languages we support UI-wise (display names)
LANG_CANDIDATES = ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi"]
detected_languages = {}  # e.g. {"English": {"query": "Query_English", ...}, ...}

norm_cols = list(normalized_map.keys())

for lang in LANG_CANDIDATES:
    lang_key = lang.lower()
    # patterns to match normalized names
    q_candidates = [f"query_{lang_key}", f"query{lang_key}", f"q_{lang_key}", f"question_{lang_key}", f"question{lang_key}"]
    s_candidates = [f"short_{lang_key}", f"short{lang_key}", f"ans_{lang_key}", f"answer_{lang_key}"]
    d_candidates = [f"detailed_{lang_key}", f"detailed{lang_key}", f"detail_{lang_key}", f"d_{lang_key}"]

    q = next((normalized_map[n] for n in norm_cols if n in q_candidates), None)
    s = next((normalized_map[n] for n in norm_cols if n in s_candidates), None)
    d = next((normalized_map[n] for n in norm_cols if n in d_candidates), None)

    # fallback: contains lang and startswith query/short/detailed
    if q is None:
        q = next((normalized_map[n] for n in norm_cols if n.startswith("query") and lang_key in n), None)
    if s is None:
        s = next((normalized_map[n] for n in norm_cols if n.startswith("short") and lang_key in n), None)
    if d is None:
        d = next((normalized_map[n] for n in norm_cols if (n.startswith("detailed") or n.startswith("detail") or n.startswith("d_")) and lang_key in n), None)

    if q or s or d:
        detected_languages[lang] = {"query": q, "short": s, "detailed": d}

# If nothing detected, show manual mapping UI
if not detected_languages:
    st.sidebar.markdown("### Column mapping required")
    st.sidebar.info("Auto-detection did not find Query_<Language> columns. Please map columns manually.")
    chosen = st.sidebar.selectbox("Map language label", LANG_CANDIDATES)
    all_cols = df.columns.tolist()
    mq = st.sidebar.selectbox(f"Select Query column for {chosen}", options=["-- none --"] + all_cols, index=0)
    ms = st.sidebar.selectbox(f"Select Short column for {chosen}", options=["-- none --"] + all_cols, index=0)
    md = st.sidebar.selectbox(f"Select Detailed column for {chosen}", options=["-- none --"] + all_cols, index=0)
    if mq != "-- none --":
        detected_languages[chosen] = {"query": mq, "short": (ms if ms != "-- none --" else None), "detailed": (md if md != "-- none --" else None)}
    else:
        st.error("Please map at least the Query column. App cannot continue without it.")
        st.stop()

# Build available language list and allow selection (sidebar)
available_languages = list(detected_languages.keys())
st.sidebar.markdown("### Language")
selected_lang = st.sidebar.selectbox("Choose language", available_languages, index=0)

# resolve column names
col_query = detected_languages[selected_lang]["query"]
col_short = detected_languages[selected_lang]["short"]
col_detailed = detected_languages[selected_lang]["detailed"]

if col_query is None:
    st.error(f"Selected language {selected_lang} has no Query column mapped. Please remap in sidebar.")
    st.stop()

# -------------------------------
# RAG: embedding model + corpus embeddings
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
def build_corpus_and_embeddings(query_series):
    """Return (queries_list, embeddings_array_or_None). embeddings may be None if model missing."""
    queries = query_series.dropna().astype(str).tolist()
    if embedding_model is None or not queries:
        return queries, None
    try:
        emb = embedding_model.encode(queries, show_progress_bar=False)
        return queries, emb
    except Exception:
        return queries, None

queries_list, corpus_embeddings = build_corpus_and_embeddings(df[col_query] if col_query in df.columns else pd.Series([], dtype=object))

# -------------------------------
# UI Header
# -------------------------------
st.markdown("<div class='main-header'><span>⚖️</span><h1>Nyayasetu — AI Legal Consultant (RAG)</h1></div>", unsafe_allow_html=True)
st.markdown("#### Quick multi-language legal guidance (RAG-enabled).")
st.markdown("---")

# show some dataset info
st.info(f"Loaded sources: {', '.join(detected_sources)} — entries: {len(df)}")
st.caption(f"Using language: **{selected_lang}** — Query column: `{col_query}`")

# -------------------------------
# Input area
# -------------------------------
st.markdown("### Ask your legal question")
user_q = st.text_input("Enter your question", value="", key="user_question_input", placeholder="e.g., Can my landlord evict me without notice?")
use_rag = st.checkbox("Use RAG semantic search (recommended)", value=(embedding_model is not None and corpus_embeddings is not None), disabled=(embedding_model is None or corpus_embeddings is None))

if st.button("Get Answer"):
    if not user_q or user_q.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            matched_row = None
            confidence = 0.0

            # RAG path
            if use_rag and corpus_embeddings is not None:
                try:
                    q_emb = embedding_model.encode([user_q])
                    scores = np.dot(corpus_embeddings, q_emb.T).flatten()
                    top_idx = int(np.argmax(scores))
                    top_score = float(scores[top_idx])
                    # use a conservative threshold for relevance in embedding-space (empirical)
                    if top_score > 0.25:  # tuneable
                        matched_row = df.iloc[top_idx]
                        confidence = float(top_score) * 100.0
                        st.success(f"✅ RAG match found (score: {top_score:.3f})")
                except Exception:
                    matched_row = None

            # Fallback fuzzy matching if RAG not used or failed
            if matched_row is None and FUZZY_AVAILABLE and queries_list:
                try:
                    best_match = process.extractOne(user_q, queries_list, scorer=fuzz.WRatio)
                    if best_match:
                        match_text, score, pos = best_match
                        if score >= 50:
                            # find the row for this matched query text
                            idx = df[df[col_query].astype(str) == match_text].index
                            if not idx.empty:
                                matched_row = df.loc[idx[0]]
                                confidence = float(score)
                                st.success(f"✅ Fuzzy match found (similarity: {score:.1f}%)")
                except Exception:
                    matched_row = None

            # Prepare answers
            if matched_row is not None:
                short_ans = matched_row.get(col_short, "Short answer not available")
                detailed_ans = matched_row.get(col_detailed, short_ans if col_detailed is None else "Detailed answer not available")
            else:
                short_ans = "❌ No relevant answer found. Try rephrasing your question or consult a legal professional."
                detailed_ans = short_ans
                st.warning("No sufficiently relevant match found.")

        # Display Answers
        st.markdown("---")
        st.markdown("#### Short Answer")
        st.info(str(short_ans))
        st.markdown("#### Detailed Answer")
        with st.expander("View full details", expanded=False):
            st.write(str(detailed_ans))

        # TTS playback for short answer
        try:
            lang_code_map = { "English":"en", "Hindi":"hi", "Bengali":"bn", "Tamil":"ta", "Telugu":"te", "Marathi":"mr" }
            lang_code = lang_code_map.get(selected_lang, "en")
            tts = gTTS(text=str(short_ans), lang=lang_code, slow=False)
            audio_buf = BytesIO()
            tts.write_to_fp(audio_buf)
            audio_buf.seek(0)
            st.audio(audio_buf.read(), format="audio/mp3")
        except Exception as e:
            st.warning(f"Audio generation failed: {e}")

        # Add to session chat history
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": user_q,
            "answer": str(short_ans),
            "language": selected_lang,
            "confidence": float(confidence)
        })

# -------------------------------
# Recent history & footer
# -------------------------------
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### Recent queries")
    for entry in reversed(st.session_state.chat_history[-8:]):
        st.write(f"Q: {entry['question']}")
        st.write(f"A: {entry['answer']}")
        st.caption(f"{entry['timestamp']} | {entry['language']} | conf: {entry.get('confidence',0):.1f}")
        st.markdown("---")

st.markdown("---")
st.caption("⚠️ Disclaimer: This assistant provides general informational guidance only and is not a substitute for professional legal advice.")

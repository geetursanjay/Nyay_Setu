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

# Translation libs
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
    try:
        from deep_translator import GoogleTranslator as DeepGoogleTranslator
        GoogleTranslator = DeepGoogleTranslator
        TRANSLATION_AVAILABLE = True
        TRANSLATOR_TYPE = "deep"
    except Exception:
        TRANSLATION_AVAILABLE = False
        TRANSLATOR_TYPE = None

# ---------------------------------------
# PAGE CONFIG + CSS
# ---------------------------------------
st.set_page_config(layout="wide", page_title="Nyayasetu — Legal Assistant")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');

    .stApp {
        background-image: url("https://raw.githubusercontent.com/geetursanjay/Nyay_Setu/main/background.png");
        background-size: cover;
        background-repeat: no-repeat;
    }
    .stApp::before {
        content:"";
        position:absolute;
        top:0; left:0;
        width:100%; height:100%;
        background-color:rgba(0,0,0,0.45);
        z-index:-1;
    }
    .main-header {
        display:flex;
        align-items:center;
        justify-content:center;
        gap:12px;
        font-family:'Dancing+Script', cursive;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------
# HEADER
# ---------------------------------------
st.markdown("""
<div class='main-header'>
    <span style="font-size:48px;">⚖️</span>
    <div>
        <h1 style="margin:0; font-size:40px; color:#0b3d91;">Nyayasetu — Legal Assistant</h1>
        <p style="margin:0; color:rgba(255,255,255,0.8);">
            AI-powered multilingual legal guidance (Title → Summary → Case)
        </p>
    </div>
</div>
<hr style="opacity:0.3;">
""", unsafe_allow_html=True)

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
@st.cache_data
def load_csv_candidates():
    files = ["train.csv.gz", "test.csv.gz", "train.csv", "test.csv"]
    df_list, src, errors = [], [], []

    def try_read(path):
        for enc in ["utf-8", "latin1", "cp1252", "ISO-8859-1"]:
            try:
                df = pd.read_csv(path, encoding=enc, compression="infer")
                return df, None
            except Exception as e:
                last_error = e
        return None, last_error

    for f in files:
        if os.path.exists(f):
            df, err = try_read(f)
            if df is not None:
                df.columns = [str(c) for c in df.columns]
                df["_source"] = f
                df_list.append(df)
                src.append(f)
            else:
                errors.append(f"{f}: {err}")

    if not df_list:
        return pd.DataFrame(), [], errors

    df_final = pd.concat(df_list, ignore_index=True)
    return df_final, src, errors


df, detected_sources, load_errors = load_csv_candidates()

if not detected_sources:
    st.error("Dataset missing. Place train.csv.gz or test.csv.gz in the folder.")
    st.stop()

st.info(f"Loaded: {', '.join(detected_sources)} • Rows: {len(df)}")

if load_errors:
    for e in load_errors:
        st.warning(str(e))

# ---------------------------------------
# COLUMN MAPPING (Title / Summary / Case)
# ---------------------------------------
cols = df.columns.tolist()

def detect(colname):
    return next((c for c in cols if c.lower() == colname.lower()), None)

col_query = detect("Title")
col_short = detect("Summary")
col_detailed = detect("Case")

# fallback heuristics
if col_query is None:
    for p in ["title", "question", "query", "subject"]:
        col_query = next((c for c in cols if p in c.lower()), None)
        if col_query: break

if col_short is None:
    for p in ["summary", "short", "answer"]:
        col_short = next((c for c in cols if p in c.lower()), None)
        if col_short: break

if col_detailed is None:
    for p in ["case", "details", "description", "full"]:
        col_detailed = next((c for c in cols if p in c.lower()), None)
        if col_detailed: break

if col_short is None:  col_short = col_query
if col_detailed is None: col_detailed = col_short

if col_query is None:
    st.error("No Query column found (Title, Question etc.)")
    st.stop()

# ---------------------------------------
# LANGUAGE SETTINGS
# ---------------------------------------
language_map = {
    "English": "en", "Hindi": "hi", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te", "Marathi": "mr"
}

st.sidebar.subheader("Language")
selected_language = st.sidebar.selectbox("Choose language:", list(language_map.keys()))

# Summarization options
st.sidebar.subheader("Summary Options")
auto_summary = st.sidebar.checkbox("Auto-summarize answers", True)
short_sentences = st.sidebar.slider("Short Answer sentences", 1, 6, 2)
detailed_sentences = st.sidebar.slider("Detailed Summary sentences", 2, 12, 4)

# ---------------------------------------
# Sentence-based summarizer
# ---------------------------------------
splitter = re.compile(r"(?<=[.!?])\s+")

def summarize(text, n):
    if not text:
        return text
    sentences = splitter.split(text.strip())
    if len(sentences) <= n:
        return text
    return " ".join(sentences[:n])

# ---------------------------------------
# TRANSLATION
# ---------------------------------------
def translate(text, lang):
    if not TRANSLATION_AVAILABLE or lang == "en":
        return text
    try:
        if TRANSLATOR_TYPE == "googletrans":
            res = translator_google.translate(text, dest=lang)
            return res.text
        if TRANSLATOR_TYPE == "deep":
            return GoogleTranslator(source="auto", target=lang).translate(text)
    except:
        return text
    return text

# ---------------------------------------
# RAG MODEL
# ---------------------------------------
@st.cache_resource
def load_rag():
    if EMBEDDINGS_AVAILABLE:
        try:
            return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except:
            return None
    return None

rag_model = load_rag()

@st.cache_data
def build_embeds():
    queries = df[col_query].dropna().astype(str).tolist()
    if rag_model is None:
        return queries, None
    try:
        emb = rag_model.encode(queries, show_progress_bar=False)
        return queries, emb
    except:
        return queries, None

queries_list, embeddings = build_embeds()

# ---------------------------------------
# MAIN UI — SEARCH
# ---------------------------------------
st.markdown("### Instant Legal Guidance")
st.markdown("Try suggested queries:")

examples = df[col_query].dropna().astype(str).tolist()[:3]
if examples:
    ecols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if ecols[i].button(ex[:60] + ("..." if len(ex) > 60 else "")):
            st.session_state['q'] = ex

query = st.text_input("Enter your question:", st.session_state.get("q", ""))

use_rag = st.checkbox("Use RAG semantic search", value=(embeddings is not None))

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Searching…"):
            matched_row = None
            confidence = 0

            # RAG search
            if use_rag and embeddings is not None and rag_model is not None:
                try:
                    q_emb = rag_model.encode([query])
                    scores = np.dot(embeddings, q_emb.T).flatten()
                    idx = int(np.argmax(scores))
                    if scores[idx] > 0.25:
                        matched_row = df.iloc[idx]
                        confidence = float(scores[idx]) * 100
                        st.success(f"RAG match ({confidence:.2f}%)")
                except:
                    pass

            # Fuzzy fallback
            if matched_row is None and FUZZY_AVAILABLE:
                best = process.extractOne(query, queries_list, scorer=fuzz.WRatio)
                if best:
                    txt, score, pos = best
                    if score >= 50:
                        matched_row = df.loc[df[col_query] == txt].iloc[0]
                        confidence = score
                        st.success(f"Fuzzy match ({score}%)")

            # Final answers
            if matched_row is None:
                short = "No relevant answer found."
                detailed = short
            else:
                short = str(matched_row[col_short])
                detailed = str(matched_row[col_detailed])

        # Summaries
        if auto_summary:
            short_s = summarize(short, short_sentences)
            detailed_s = summarize(detailed, detailed_sentences)
        else:
            short_s = short
            detailed_s = detailed

        # Translate short summary
        lang = language_map[selected_language]
        short_trans = translate(short_s, lang)
        detailed_trans = translate(detailed_s, lang)

        # ------------------------
        # OUTPUT
        # ------------------------
        st.markdown("---")
        st.markdown("### Short Answer")
        st.info(short_trans)

        st.markdown("### Detailed Summary")
        with st.expander("Show summarized details"):
            st.write(detailed_trans)

        with st.expander("Full detailed case"):
            st.write(detailed)

        # TTS
        try:
            tts = gTTS(short_trans, lang=lang)
            buf = BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            st.audio(buf.read(), format="audio/mp3")
        except:
            st.warning("Audio unavailable.")

# ---------------------------------------
# Lawyer Assistance — Coming Soon
# ---------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="
        padding:18px;
        background:rgba(255,255,255,0.85);
        border-left:6px solid #0b3d91;
        border-radius:10px;
        margin-top:20px;
    ">
        <h3 style="margin:0; color:#0b3d91;">👨‍⚖️ Lawyer Assistance</h3>
        <p style="margin:6px 0 0; color:#333;">
            We are building a feature to help you connect with verified lawyers for personalised legal consultation.<br>
            <strong>Coming soon… 🚀</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------
# FOOTER
# ---------------------------------------
st.markdown("---")
st.caption("⚠️ This assistant provides general legal information only. Not a substitute for professional legal advice.")

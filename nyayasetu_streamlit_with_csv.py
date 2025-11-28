import streamlit as st
import pandas as pd
import numpy as np
from gtts import gTTS
from io import BytesIO
from datetime import datetime
import os

# Optional deps
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
# Page config
# -------------------------------
st.set_page_config(layout="wide", page_title="Nyayasetu - AI Legal Assistant")

# -------------------------------
# CSS (optional)
# -------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');
    .main-header { display:flex; justify-content:center; align-items:center; gap:12px; font-family:'Dancing Script', cursive; }
    .main-header h1{ margin:0; color:#0b3d91; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Session state init
# -------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
if 'user_question' not in st.session_state:
    st.session_state.user_question = ''
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = False

# -------------------------------
# Data loader (cache-safe)
# -------------------------------
@st.cache_data
def load_data_gz():
    """
    Load train/test CSV files, accepting compressed .gz (or plain .csv).
    Returns (df_combined, sources_list, errors_list)
    """
    files_to_try = ["train.csv.gz", "test.csv.gz", "train.csv", "test.csv"]
    df_list = []
    sources = []
    errors = []

    def try_read(path):
        # Let pandas infer compression; try a few encodings
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
        last_exc = None
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, compression='infer'), None
            except Exception as e:
                last_exc = e
        return None, str(last_exc)

    for fname in files_to_try:
        if os.path.exists(fname):
            df_loaded, err = try_read(fname)
            if df_loaded is not None:
                df_loaded.columns = [str(c) for c in df_loaded.columns]
                df_loaded['_source'] = fname
                df_list.append(df_loaded)
                sources.append(fname)
            else:
                errors.append(f"{fname}: {err}")

    if not df_list:
        return pd.DataFrame(), [], errors

    # Concatenate aligning columns
    try:
        df_combined = pd.concat(df_list, ignore_index=True, sort=False)
    except Exception as e:
        return pd.DataFrame(), sources, [f"Concatenation failed: {e}"]

    return df_combined.reset_index(drop=True), sources, errors

# Load dataset (caller shows UI errors)
df, detected_sources, load_errors = load_data_gz()

# Show loader messages after cached call
if load_errors:
    for err in load_errors:
        st.warning(f"⚠️ {err}")

if not detected_sources:
    st.info("No dataset files found. Please upload `train.csv.gz` and/or `test.csv.gz` in Admin Mode.")

# -------------------------------
# Embedding model loader (optional)
# -------------------------------
@st.cache_resource
def get_embedding_model():
    if EMBEDDINGS_AVAILABLE:
        try:
            return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception:
            return None
    return None

embedding_model = get_embedding_model()

def create_embeddings(texts):
    if embedding_model is not None and len(texts) > 0:
        try:
            return embedding_model.encode(texts, show_progress_bar=False)
        except Exception:
            return None
    return None

def semantic_search(query, texts, top_k=3):
    if embedding_model is None or not texts:
        return []
    try:
        q_emb = embedding_model.encode([query])
        corpus = create_embeddings(texts)
        if corpus is None:
            return []
        scores = np.dot(corpus, q_emb.T).flatten()
        idxs = np.argsort(scores)[-top_k:][::-1]
        return [{'index': int(i), 'score': float(scores[i]), 'text': texts[i]} for i in idxs]
    except Exception:
        return []

# -------------------------------
# Language detection from dataset
# -------------------------------
language_map = {
    "English": "en", "Hindi": "hi", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te", "Marathi": "mr"
}
available_languages = []
if not df.empty:
    for lang in language_map.keys():
        if f"Query_{lang}" in df.columns:
            available_languages.append(lang)

# -------------------------------
# Header & dataset info
# -------------------------------
st.markdown("<div class='main-header'><span>⚖️</span><h1>Nyayasetu - AI Legal Consultant</h1></div>", unsafe_allow_html=True)
st.markdown("---")
if detected_sources:
    st.success(f"Loaded: {', '.join(detected_sources)} (entries: {len(df)})")
else:
    st.info("No compressed dataset loaded yet.")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("Settings")
    if available_languages:
        selected_lang_display = st.selectbox("Language", available_languages)
        col_query = f"Query_{selected_lang_display}"
        col_short = f"Short_{selected_lang_display}"
        col_detailed = f"Detailed_{selected_lang_display}"
    else:
        selected_lang_display = "English"
        col_query = "Query_English"
        col_short = "Short_English"
        col_detailed = "Detailed_English"

    st.subheader("Mode")
    mode = st.radio("Choose mode:", ["User Mode", "Admin Mode"])
    st.session_state.admin_mode = (mode == "Admin Mode")

    st.markdown("---")
    st.subheader("System status")
    st.write("RAG embeddings available" if EMBEDDINGS_AVAILABLE else "RAG embeddings NOT available")
    st.write("Fuzzy matching available" if FUZZY_AVAILABLE else "Fuzzy matching NOT available")
    st.markdown("---")
    st.metric("Entries", len(df))
    st.metric("Languages", len(available_languages))
    st.metric("Chat history", len(st.session_state.chat_history))

# -------------------------------
# Admin Mode (upload accepts .gz)
# -------------------------------
if st.session_state.admin_mode:
    st.markdown("## Admin: Upload train/test (compressed)")
    st.info("Only `train.csv.gz` and `test.csv.gz` are expected. You may also upload plain `.csv` — it will be saved as .csv.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_train = st.file_uploader("Upload train.csv.gz", type=["gz", "csv"], key="up_train")
        if uploaded_train is not None:
            try:
                # decide filename
                save_name = "train.csv.gz" if uploaded_train.name.endswith(".gz") else "train.csv"
                with open(save_name, "wb") as f:
                    f.write(uploaded_train.getbuffer())
                st.success(f"Saved {save_name}")
                st.dataframe(pd.read_csv(save_name, compression='infer', nrows=5))
                if st.button("Reload data (after upload)", key="reload_after_train"):
                    st.cache_data.clear()
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    with col2:
        uploaded_test = st.file_uploader("Upload test.csv.gz", type=["gz", "csv"], key="up_test")
        if uploaded_test is not None:
            try:
                save_name = "test.csv.gz" if uploaded_test.name.endswith(".gz") else "test.csv"
                with open(save_name, "wb") as f:
                    f.write(uploaded_test.getbuffer())
                st.success(f"Saved {save_name}")
                st.dataframe(pd.read_csv(save_name, compression='infer', nrows=5))
                if st.button("Reload data (after upload)", key="reload_after_test"):
                    st.cache_data.clear()
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.markdown("### Add a new entry (appends to train.csv.gz as plain CSV fallback)")
    if not available_languages:
        st.warning("Upload datasets first to detect languages.")
    else:
        with st.form("add_form"):
            new_row = {}
            for lang in available_languages:
                st.subheader(f"{lang}")
                colA, colB, colC = st.columns(3)
                with colA:
                    new_row[f"Query_{lang}"] = st.text_input(f"Query ({lang})", key=f"aq_{lang}")
                with colB:
                    new_row[f"Short_{lang}"] = st.text_area(f"Short ({lang})", key=f"as_{lang}")
                with colC:
                    new_row[f"Detailed_{lang}"] = st.text_area(f"Detailed ({lang})", key=f"ad_{lang}")
            add_sub = st.form_submit_button("Add to train.csv (appends)")
            if add_sub:
                try:
                    row_df = pd.DataFrame([new_row])
                    # if train.csv.gz exists, append to train.csv (we'll write as plain CSV fallback)
                    target = "train.csv"
                    if os.path.exists("train.csv.gz") and not os.path.exists("train.csv"):
                        # if only gz exists, create train.csv by reading gz then appending
                        base = pd.read_csv("train.csv.gz", compression='infer')
                        base = pd.concat([base, row_df], ignore_index=True, sort=False)
                        base.to_csv("train.csv", index=False)
                    elif os.path.exists("train.csv"):
                        base = pd.read_csv("train.csv")
                        base = pd.concat([base, row_df], ignore_index=True, sort=False)
                        base.to_csv("train.csv", index=False)
                    else:
                        row_df.to_csv("train.csv", index=False)
                    st.success("Appended new row to train.csv (use reload to refresh app).")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"Failed to add: {e}")

    st.markdown("---")
    st.markdown("### Export / Preview")
    if st.button("Download combined as CSV"):
        try:
            csv_out = df.drop(columns=['_source'], errors='ignore').to_csv(index=False)
            st.download_button("Download CSV", csv_out, file_name=f"nyayasetu_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Export failed: {e}")

    if st.checkbox("Show dataset preview"):
        st.dataframe(df.drop(columns=['_source'], errors='ignore'), use_container_width=True)

# -------------------------------
# User Mode: Query / Answer
# -------------------------------
else:
    st.markdown("## Ask a legal question")
    if df.empty or not detected_sources:
        st.warning("No dataset loaded. Switch to Admin Mode to upload `train.csv.gz` / `test.csv.gz`.")
        st.stop()
    if not available_languages:
        st.error("Dataset loaded but no Query_<Language> columns detected.")
        st.stop()

    # Example buttons
    example_queries = df[col_query].dropna().astype(str).tolist()[:3]
    if example_queries:
        ex_cols = st.columns(len(example_queries))
        for i, q in enumerate(example_queries):
            with ex_cols[i]:
                if st.button(q[:50] + ("..." if len(q) > 50 else ""), key=f"ex_{i}"):
                    st.session_state.user_question = q
                    st.experimental_rerun()

    user_question = st.text_input("Enter your legal question:", value=st.session_state.user_question, key="uq")
    colA, colB = st.columns([3,1])
    with colA:
        ask = st.button("Get Answer", use_container_width=True, type="primary")
    with colB:
        use_rag = st.checkbox("Use RAG", value=(embedding_model is not None), disabled=(embedding_model is None))

    if ask and user_question:
        with st.spinner("Searching..."):
            queries = df[col_query].dropna().astype(str).tolist()
            matched = None
            conf = 0.0

            # RAG semantic search
            if use_rag and EMBEDDINGS_AVAILABLE and embedding_model is not None:
                results = semantic_search(user_question, queries, top_k=3)
                if results and results[0]['score'] > 0.3:
                    idx = results[0]['index']
                    conf = results[0]['score'] * 100
                    matched = df.iloc[idx]

            # Fuzzy fallback
            if matched is None and FUZZY_AVAILABLE:
                try:
                    best_match = process.extractOne(user_question, queries, scorer=fuzz.WRatio, score_cutoff=40)
                    if best_match:
                        match_text, score, pos = best_match
                        matched = df[df[col_query] == match_text].iloc[0]
                        conf = score
                except Exception:
                    pass

            if matched is not None:
                short_answer = str(matched.get(col_short, "Answer not available"))
                detailed_answer = str(matched.get(col_detailed, "Detailed not available"))
                st.success(f"Found answer (confidence: {conf:.2f}%)")
            else:
                short_answer = "❌ No relevant answer found. Please rephrase or contact an expert."
                detailed_answer = short_answer
                st.warning("No sufficiently relevant match found.")

            st.markdown("---")
            left, right = st.columns(2)
            with left:
                st.markdown("#### Short answer")
                st.info(short_answer)
            with right:
                st.markdown("#### Detailed answer")
                with st.expander("View full details", expanded=True):
                    st.write(detailed_answer)

            # TTS
            try:
                lang_code = language_map.get(selected_lang_display, "en")
                tts = gTTS(text=short_answer, lang=lang_code, slow=False)
                buf = BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                st.audio(buf.read(), format="audio/mp3")
            except Exception as e:
                st.warning(f"Audio generation failed: {e}")

            # Save chat history
            st.session_state.chat_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": user_question,
                "answer": short_answer,
                "language": selected_lang_display,
                "confidence": float(conf)
            })

            # Feedback
            st.markdown("---")
            st.markdown("Was this helpful?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍 Yes", key="fb_yes"):
                    st.success("Thanks!")
            with c2:
                if st.button("👎 No", key="fb_no"):
                    st.session_state.show_feedback = True

            if st.session_state.show_feedback:
                feedback = st.text_area("Please tell us how to improve", key="fb_text")
                if st.button("Submit feedback", key="fb_submit"):
                    st.info("Thanks for your feedback!")
                    st.session_state.show_feedback = False

    # Chat history preview
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Recent chat history")
        with st.expander("View", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
                st.write(f"Q: {chat['question']}")
                st.write(f"A: {chat['answer']}")
                st.caption(f"{chat['timestamp']} | {chat['language']} | conf: {chat.get('confidence',0):.1f}%")
                st.markdown("---")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Disclaimer: This provides general information only. Not a substitute for legal advice.")

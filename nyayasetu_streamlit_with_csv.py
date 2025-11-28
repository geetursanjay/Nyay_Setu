import streamlit as st 
import pandas as pd
from gtts import gTTS
from io import BytesIO
import os
import re
import json
from datetime import datetime
import numpy as np

# For RAG implementation
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    st.warning("⚠️ Install sentence-transformers and faiss-cpu for better semantic search")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(layout="wide", page_title="Nyayasetu - AI Legal Assistant")

# -------------------------------
# Custom CSS
# -------------------------------
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
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }
    .main-header {
        display: flex; justify-content: center; align-items: center; gap: 15px;
        font-family: 'Dancing Script', cursive; color: darkblue; text-align: center;
    }
    .main-header h1 {
        font-size: 3.5rem; font-weight: 700; text-shadow: 2px 2px 4px #FFFFFF;
    }
    .main-header .symbol {
        font-size: 3.5rem; color: #FFD700; text-shadow: 2px 2px 4px #000000;
    }
    .highlight {
        background-color: yellow;
        font-weight: bold;
    }
    .admin-section {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Initialize Session State
# -------------------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
if 'dataset_modified' not in st.session_state:
    st.session_state.dataset_modified = False

# -------------------------------
# Load Dataset (supports SIH_Dataset_Final.xlsx, train.csv, test.csv and uploaded files)
# -------------------------------
@st.cache_data
def load_data(preferred_source=None):
    """Try to load dataset from multiple possible files. If preferred_source is provided,
    prioritize it.
    Returns (df, sources_list)
    """
    sources = []
    df_list = []

    # Try Excel first
    if os.path.exists("SIH_Dataset_Final.xlsx"):
        try:
            df_x = pd.read_excel("SIH_Dataset_Final.xlsx")
            df_x['_source'] = 'SIH_Dataset_Final.xlsx'
            df_list.append(df_x)
            sources.append('SIH_Dataset_Final.xlsx')
        except Exception:
            pass

    # Try train.csv
    if os.path.exists("train.csv"):
        try:
            df_tr = pd.read_csv("train.csv")
            df_tr['_source'] = 'train.csv'
            df_list.append(df_tr)
            sources.append('train.csv')
        except Exception:
            pass

    # Try test.csv
    if os.path.exists("test.csv"):
        try:
            df_te = pd.read_csv("test.csv")
            df_te['_source'] = 'test.csv'
            df_list.append(df_te)
            sources.append('test.csv')
        except Exception:
            pass

    # If nothing found, return empty df
    if not df_list:
        return pd.DataFrame(), []

    # If preferred source specified and present, return only that
    if preferred_source and preferred_source in sources:
        for d in df_list:
            if d['_source'].iloc[0] == preferred_source:
                return d.reset_index(drop=True), sources

    # Otherwise concatenate all (inner join columns by filling missing values)
    concatenated = pd.concat(df_list, ignore_index=True, sort=False).reset_index(drop=True)
    return concatenated, sources

# Load dataset (no preferred source by default)
df, detected_sources = load_data()

# -------------------------------
# RAG System Setup
# -------------------------------
@st.cache_resource
def load_embedding_model():
    if EMBEDDINGS_AVAILABLE:
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return None

embedding_model = load_embedding_model()

def create_embeddings(texts):
    """Create embeddings for text data"""
    if embedding_model and texts:
        return embedding_model.encode(texts, show_progress_bar=False)
    return None

def semantic_search(query, texts, top_k=3):
    """Perform semantic search using embeddings"""
    if not embedding_model or not texts:
        return []
    
    query_embedding = embedding_model.encode([query])
    corpus_embeddings = create_embeddings(texts)
    
    # Calculate cosine similarity
    scores = np.dot(corpus_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'index': idx,
            'score': float(scores[idx]),
            'text': texts[idx]
        })
    return results

# -------------------------------
# Language Configuration
# -------------------------------
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr"
}

available_languages = []
for lang in language_map.keys():
    if f"Query_{lang}" in df.columns:
        available_languages.append(lang)

if not available_languages and not df.empty:
    st.error("❌ No valid language columns found in dataset.")
    st.stop()

# -------------------------------
# App Header
# -------------------------------
st.markdown(
    """
    <div class="main-header">
        <span class="symbol">⚖️</span>
        <h1>Nyayasetu - AI Legal Consultant</h1>
    </div>
    <p style='text-align:center; color: #000000; text-shadow: 1px 1px 2px #FFFFFF; font-weight: italic;'>
    AI-powered legal assistant with RAG technology for accurate, context-aware legal guidance in multiple languages.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Allow user to pick which source to use if multiple detected
    if detected_sources:
        preferred = st.selectbox("Dataset source:", options=[None] + detected_sources, format_func=lambda x: 'All sources' if x is None else x)
        if preferred:
            df, _ = load_data(preferred_source=preferred)
        # Recompute available_languages based on selected df
        available_languages = []
        for lang in language_map.keys():
            if f"Query_{lang}" in df.columns:
                available_languages.append(lang)

    if available_languages:
        selected_lang_display = st.selectbox("Select language:", available_languages)
        col_query = f"Query_{selected_lang_display}"
        col_short = f"Short_{selected_lang_display}"
        col_detailed = f"Detailed_{selected_lang_display}"
    
    st.markdown("---")
    st.subheader("🔧 System Mode")
    mode = st.radio("Choose mode:", ["User Mode", "Admin Mode"])
    st.session_state.admin_mode = (mode == "Admin Mode")
    
    st.markdown("---")
    st.subheader("📊 Statistics")
    if not df.empty:
        st.metric("Total Q&A Pairs", len(df))
        st.metric("Languages", len(available_languages))
    st.metric("Chat History", len(st.session_state.chat_history))
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------
# ADMIN MODE: Dataset Management
# -------------------------------
if st.session_state.admin_mode:
    st.markdown("## 🔐 Admin Panel - Dataset Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add New Entry", "✏️ Edit Existing", "📥 Export Dataset", "📤 Upload CSV Files"])
    
    with tab4:
        st.markdown("### Upload train.csv / test.csv")
        st.info("You can upload new train.csv or test.csv files. They will be saved to the app folder and included in the dataset sources.")
        uploaded_train = st.file_uploader("Upload train.csv", type=["csv"], key="upload_train")
        uploaded_test = st.file_uploader("Upload test.csv", type=["csv"], key="upload_test")
        if uploaded_train is not None:
            try:
                df_tr = pd.read_csv(uploaded_train)
                df_tr.to_csv("train.csv", index=False)
                st.success("train.csv uploaded and saved.")
                st.cache_data.clear()
                df, detected_sources = load_data()
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to save train.csv: {e}")
        if uploaded_test is not None:
            try:
                df_te = pd.read_csv(uploaded_test)
                df_te.to_csv("test.csv", index=False)
                st.success("test.csv uploaded and saved.")
                st.cache_data.clear()
                df, detected_sources = load_data()
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to save test.csv: {e}")
    
    with tab1:
        st.markdown("### Add New Q&A Entry")
        with st.form("add_entry_form"):
            new_entry = {}
            
            for lang in available_languages:
                st.subheader(f"{lang} Content")
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_entry[f"Query_{lang}"] = st.text_input(f"Query ({lang})", key=f"q_{lang}")
                with col2:
                    new_entry[f"Short_{lang}"] = st.text_area(f"Short Answer ({lang})", key=f"s_{lang}", height=100)
                with col3:
                    new_entry[f"Detailed_{lang}"] = st.text_area(f"Detailed Answer ({lang})", key=f"d_{lang}", height=100)
            
            submitted = st.form_submit_button("➕ Add to Dataset")
            
            if submitted:
                new_row = pd.DataFrame([new_entry])
                # if there is an excel file, append to it, else create/append to train.csv
                try:
                    if os.path.exists("SIH_Dataset_Final.xlsx"):
                        df_excel = pd.read_excel("SIH_Dataset_Final.xlsx")
                        df_excel = pd.concat([df_excel, new_row], ignore_index=True)
                        df_excel.to_excel("SIH_Dataset_Final.xlsx", index=False)
                    else:
                        # append to train.csv if exists otherwise create train.csv
                        if os.path.exists("train.csv"):
                            df_tr = pd.read_csv("train.csv")
                            df_tr = pd.concat([df_tr, new_row], ignore_index=True)
                            df_tr.to_csv("train.csv", index=False)
                        else:
                            new_row.to_csv("train.csv", index=False)
                    st.success("✅ New entry added successfully!")
                    st.session_state.dataset_modified = True
                    st.cache_data.clear()
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to save new entry: {e}")
    
    with tab2:
        st.markdown("### Edit Existing Entries")
        if not df.empty:
            entry_index = st.selectbox("Select entry to edit:", range(len(df)), 
                                      format_func=lambda x: f"Entry {x+1}: {str(df.iloc[x].get(col_query, 'N/A'))[:50]}...")
            
            if entry_index is not None:
                with st.form("edit_entry_form"):
                    edited_entry = {}
                    
                    for lang in available_languages:
                        st.subheader(f"{lang} Content")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            edited_entry[f"Query_{lang}"] = st.text_input(
                                f"Query ({lang})", 
                                value=str(df.iloc[entry_index].get(f"Query_{lang}", "")),
                                key=f"eq_{lang}"
                            )
                        with col2:
                            edited_entry[f"Short_{lang}"] = st.text_area(
                                f"Short Answer ({lang})", 
                                value=str(df.iloc[entry_index].get(f"Short_{lang}", "")),
                                height=100,
                                key=f"es_{lang}"
                            )
                        with col3:
                            edited_entry[f"Detailed_{lang}"] = st.text_area(
                                f"Detailed Answer ({lang})", 
                                value=str(df.iloc[entry_index].get(f"Detailed_{lang}", "")),
                                height=100,
                                key=f"ed_{lang}"
                            )
                    
                    col_save, col_delete = st.columns(2)
                    with col_save:
                        save_btn = st.form_submit_button("💾 Save Changes")
                    with col_delete:
                        delete_btn = st.form_submit_button("🗑️ Delete Entry", type="secondary")
                    
                    if save_btn:
                        for key, value in edited_entry.items():
                            df.at[entry_index, key] = value
                        # Persist change back to original source if possible
                        src = df.at[entry_index, '_source'] if '_source' in df.columns else None
                        try:
                            if src == 'SIH_Dataset_Final.xlsx' and os.path.exists(src):
                                df_full = pd.read_excel(src)
                                for k in edited_entry.keys():
                                    df_full.at[entry_index, k] = edited_entry[k]
                                df_full.to_excel(src, index=False)
                            elif src in ['train.csv', 'test.csv'] and os.path.exists(src):
                                df_full = pd.read_csv(src)
                                for k in edited_entry.keys():
                                    df_full.at[entry_index, k] = edited_entry[k]
                                df_full.to_csv(src, index=False)
                            else:
                                # fallback: write to train.csv
                                df.to_csv('train.csv', index=False)
                            st.success("✅ Entry updated successfully!")
                            st.cache_data.clear()
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to save edits: {e}")
                    
                    if delete_btn:
                        src = df.at[entry_index, '_source'] if '_source' in df.columns else None
                        try:
                            df = df.drop(entry_index).reset_index(drop=True)
                            # persist to the source (simple approach: overwrite train.csv)
                            df.to_csv('train.csv', index=False)
                            st.success("✅ Entry deleted successfully!")
                            st.cache_data.clear()
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to delete entry: {e}")
    
    with tab3:
        st.markdown("### Export Dataset")
        st.info(f"📊 Current dataset contains {len(df)} entries")
        
        if st.button("📥 Download Dataset as Excel"):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            
            st.download_button(
                label="⬇️ Download Excel File",
                data=output,
                file_name=f"nyayasetu_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        if st.button("📥 Download Dataset as JSON"):
            json_data = df.to_json(orient='records', force_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Download JSON File",
                data=json_data,
                file_name=f"nyayasetu_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# -------------------------------
# USER MODE: Legal Consultation
# -------------------------------
else:
    st.markdown("### 💬 Get Legal Guidance Instantly")
    
    # Example Queries
    if not df.empty and available_languages:
        st.markdown("**Try these example questions:**")
        example_queries = df[col_query].dropna().tolist()[:3]
        
        if example_queries:
            cols = st.columns(len(example_queries))
            for i, query in enumerate(example_queries):
                with cols[i]:
                    if st.button(query[:50] + "...", key=f"example_{i}"):
                        st.session_state['user_question'] = query
                        st.rerun()
    
    # User Input
    user_question = st.text_input(
        "🔍 Enter your legal question:",
        value=st.session_state.get('user_question', ''),
        key='user_question_input',
        placeholder="e.g., What are my rights as a tenant?"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submitted = st.button("🔎 Get Answer", type="primary", use_container_width=True)
    with col2:
        use_rag = st.checkbox("Use RAG", value=EMBEDDINGS_AVAILABLE, disabled=not EMBEDDINGS_AVAILABLE)
    
    # Process Query
    if submitted and user_question and not df.empty:
        with st.spinner("🔍 Searching for your answer..."):
            queries = df[col_query].dropna().tolist()
            
            if use_rag and EMBEDDINGS_AVAILABLE:
                # Use semantic search
                results = semantic_search(user_question, queries, top_k=3)
                if results and results[0]['score'] > 0.3:  # Threshold for relevance
                    best_idx = results[0]['index']
                    score = results[0]['score'] * 100
                    matched_row = df.iloc[best_idx]
                    short_answer = matched_row[col_short]
                    detailed_answer = matched_row[col_detailed]
                    st.success(f"✅ Found relevant answer (Confidence: {score:.2f}%)")
                else:
                    short_answer = "❌ No relevant answer found. Please rephrase or contact a legal expert."
                    detailed_answer = short_answer
                    st.warning("No sufficiently relevant match found.")
            else:
                # Fallback to fuzzy matching
                from rapidfuzz import process, fuzz
                best_match, score, index = process.extractOne(
                    user_question, queries, scorer=fuzz.WRatio, score_cutoff=40
                )
                
                if best_match:
                    matched_row = df[df[col_query] == best_match].iloc[0]
                    short_answer = matched_row[col_short]
                    detailed_answer = matched_row[col_detailed]
                    st.success(f"✅ Found match (Similarity: {score:.2f}%)")
                else:
                    short_answer = "❌ No close match found. Try rephrasing."
                    detailed_answer = short_answer
                    st.warning("No close match found.")
            
            # Display answers
            st.markdown("---")
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("#### 📝 Short Answer")
                st.info(short_answer)
            
            with col_right:
                st.markdown("#### 📄 Detailed Answer")
                with st.expander("View full details", expanded=True):
                    st.write(detailed_answer)
            
            # Text-to-Speech
            st.markdown("---")
            st.markdown("#### 🔊 Listen to Answer")
            try:
                lang_code = language_map.get(selected_lang_display, "en")
                tts = gTTS(text=short_answer, lang=lang_code)
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.warning(f"⚠️ Audio generation failed: {e}")
            
            # Save to chat history
            st.session_state.chat_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'question': user_question,
                'answer': short_answer,
                'language': selected_lang_display
            })
            
            # Feedback
            st.markdown("---")
            st.markdown("#### Was this helpful?")
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("👍 Yes", use_container_width=True):
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎 No", use_container_width=True):
                    feedback = st.text_area("How can we improve?", key="feedback")
                    if st.button("Submit Feedback"):
                        st.info("Feedback recorded. Thank you!")
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Chat History")
        with st.expander("View previous questions"):
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                st.markdown(f"*Answer:* {chat['answer'][:100]}...")
                st.caption(f"🕐 {chat['timestamp']} | 🌐 {chat['language']}")
                st.markdown("---")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("💼 Consult a lawyer nearby → Coming soon 🚀")
with col2:
    st.info("📱 Mobile App → In Development")
with col3:
    st.info("🎓 Legal Resources → Coming soon")

st.markdown("---")
st.caption("⚠️ **Disclaimer:** This is an AI assistant providing general legal information only. Not a substitute for professional legal advice.")
st.caption("Powered by RAG Technology | Nyayasetu © 2024")

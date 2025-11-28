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
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# For fuzzy matching fallback
try:
    from rapidfuzz import process, fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

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
if 'user_question' not in st.session_state:
    st.session_state.user_question = ''

# -------------------------------
# Load Dataset from train.csv and test.csv
# -------------------------------
@st.cache_data
def load_data():
    """Load and combine train.csv and test.csv"""
    df_list = []
    sources = []
    
    # Try to load train.csv
    if os.path.exists("train.csv"):
        try:
            df_train = pd.read_csv("train.csv")
            df_train['_source'] = 'train.csv'
            df_list.append(df_train)
            sources.append('train.csv')
        except Exception as e:
            st.warning(f"Could not load train.csv: {e}")
    
    # Try to load test.csv
    if os.path.exists("test.csv"):
        try:
            df_test = pd.read_csv("test.csv")
            df_test['_source'] = 'test.csv'
            df_list.append(df_test)
            sources.append('test.csv')
        except Exception as e:
            st.warning(f"Could not load test.csv: {e}")
    
    # If no files found, return empty dataframe
    if not df_list:
        st.error("❌ No dataset files found. Please upload train.csv or test.csv")
        return pd.DataFrame(), []
    
    # Combine datasets
    df_combined = pd.concat(df_list, ignore_index=True, sort=False)
    return df_combined, sources

# Load dataset
df, detected_sources = load_data()

# -------------------------------
# RAG System Setup
# -------------------------------
@st.cache_resource
def load_embedding_model():
    if EMBEDDINGS_AVAILABLE:
        try:
            return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except Exception as e:
            st.warning(f"Could not load embedding model: {e}")
            return None
    return None

embedding_model = load_embedding_model()

def create_embeddings(texts):
    """Create embeddings for text data"""
    if embedding_model and texts:
        try:
            return embedding_model.encode(texts, show_progress_bar=False)
        except Exception:
            return None
    return None

def semantic_search(query, texts, top_k=3):
    """Perform semantic search using embeddings"""
    if not embedding_model or not texts:
        return []
    
    try:
        query_embedding = embedding_model.encode([query])
        corpus_embeddings = create_embeddings(texts)
        
        if corpus_embeddings is None:
            return []
        
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
    except Exception:
        return []

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

# Detect available languages from dataset
available_languages = []
if not df.empty:
    for lang in language_map.keys():
        if f"Query_{lang}" in df.columns:
            available_languages.append(lang)

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

# Show dataset info
if detected_sources:
    st.info(f"📊 Loaded datasets: {', '.join(detected_sources)} | Total entries: {len(df)}")
else:
    st.warning("⚠️ No datasets loaded. Please upload train.csv or test.csv in Admin Mode.")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Language selection
    if available_languages:
        selected_lang_display = st.selectbox("Select language:", available_languages)
        col_query = f"Query_{selected_lang_display}"
        col_short = f"Short_{selected_lang_display}"
        col_detailed = f"Detailed_{selected_lang_display}"
    else:
        st.warning("No languages detected in dataset")
        selected_lang_display = "English"
        col_query = "Query_English"
        col_short = "Short_English"
        col_detailed = "Detailed_English"
    
    st.markdown("---")
    st.subheader("🔧 System Mode")
    mode = st.radio("Choose mode:", ["User Mode", "Admin Mode"])
    st.session_state.admin_mode = (mode == "Admin Mode")
    
    st.markdown("---")
    st.subheader("📊 Statistics")
    st.metric("Total Q&A Pairs", len(df))
    st.metric("Languages", len(available_languages))
    st.metric("Chat History", len(st.session_state.chat_history))
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Show system status
    st.markdown("---")
    st.subheader("🔌 System Status")
    st.write("✅ RAG Embeddings" if EMBEDDINGS_AVAILABLE else "❌ RAG Embeddings")
    st.write("✅ Fuzzy Matching" if FUZZY_AVAILABLE else "❌ Fuzzy Matching")

# -------------------------------
# ADMIN MODE: Dataset Management
# -------------------------------
if st.session_state.admin_mode:
    st.markdown("## 🔐 Admin Panel - Dataset Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Dataset", "➕ Add Entry", "✏️ Edit Entry", "📥 Export Dataset"])
    
    # TAB 1: Upload CSV Files
    with tab1:
        st.markdown("### 📤 Upload Dataset Files")
        st.info("Upload your train.csv and/or test.csv files here. Expected columns: Query_[Language], Short_[Language], Detailed_[Language]")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Upload train.csv")
            uploaded_train = st.file_uploader("Choose train.csv file", type=["csv"], key="upload_train")
            if uploaded_train is not None:
                try:
                    df_train = pd.read_csv(uploaded_train)
                    df_train.to_csv("train.csv", index=False)
                    st.success("✅ train.csv uploaded successfully!")
                    st.dataframe(df_train.head())
                    if st.button("Reload Data", key="reload_train"):
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error uploading train.csv: {e}")
        
        with col2:
            st.markdown("#### Upload test.csv")
            uploaded_test = st.file_uploader("Choose test.csv file", type=["csv"], key="upload_test")
            if uploaded_test is not None:
                try:
                    df_test = pd.read_csv(uploaded_test)
                    df_test.to_csv("test.csv", index=False)
                    st.success("✅ test.csv uploaded successfully!")
                    st.dataframe(df_test.head())
                    if st.button("Reload Data", key="reload_test"):
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error uploading test.csv: {e}")
    
    # TAB 2: Add New Entry
    with tab2:
        st.markdown("### ➕ Add New Q&A Entry")
        
        if not available_languages:
            st.warning("⚠️ Please upload a dataset first to see available languages")
        else:
            with st.form("add_entry_form"):
                new_entry = {}
                
                for lang in available_languages:
                    st.subheader(f"📝 {lang} Content")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        new_entry[f"Query_{lang}"] = st.text_input(f"Query ({lang})", key=f"add_q_{lang}")
                    with col2:
                        new_entry[f"Short_{lang}"] = st.text_area(f"Short Answer ({lang})", key=f"add_s_{lang}", height=100)
                    with col3:
                        new_entry[f"Detailed_{lang}"] = st.text_area(f"Detailed Answer ({lang})", key=f"add_d_{lang}", height=100)
                
                submitted = st.form_submit_button("➕ Add to Dataset", type="primary", use_container_width=True)
                
                if submitted:
                    try:
                        new_row = pd.DataFrame([new_entry])
                        
                        # Append to train.csv
                        if os.path.exists("train.csv"):
                            df_train = pd.read_csv("train.csv")
                            df_train = pd.concat([df_train, new_row], ignore_index=True)
                            df_train.to_csv("train.csv", index=False)
                        else:
                            new_row.to_csv("train.csv", index=False)
                        
                        st.success("✅ New entry added to train.csv successfully!")
                        st.cache_data.clear()
                        st.balloons()
                        
                        if st.button("Refresh Page"):
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to add entry: {e}")
    
    # TAB 3: Edit Existing Entry
    with tab3:
        st.markdown("### ✏️ Edit Existing Entries")
        
        if df.empty:
            st.warning("⚠️ No data available to edit. Please upload a dataset first.")
        elif not available_languages:
            st.warning("⚠️ No language columns detected in dataset")
        else:
            # Select entry to edit
            entry_options = [f"Entry {i+1}: {str(df.iloc[i].get(col_query, 'N/A'))[:60]}..." for i in range(len(df))]
            selected_entry = st.selectbox("Select entry to edit:", range(len(df)), format_func=lambda x: entry_options[x])
            
            if selected_entry is not None:
                st.markdown(f"**Editing Entry {selected_entry + 1}**")
                
                with st.form("edit_entry_form"):
                    edited_entry = {}
                    
                    for lang in available_languages:
                        st.subheader(f"📝 {lang} Content")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            edited_entry[f"Query_{lang}"] = st.text_input(
                                f"Query ({lang})", 
                                value=str(df.iloc[selected_entry].get(f"Query_{lang}", "")),
                                key=f"edit_q_{lang}"
                            )
                        with col2:
                            edited_entry[f"Short_{lang}"] = st.text_area(
                                f"Short Answer ({lang})", 
                                value=str(df.iloc[selected_entry].get(f"Short_{lang}", "")),
                                height=100,
                                key=f"edit_s_{lang}"
                            )
                        with col3:
                            edited_entry[f"Detailed_{lang}"] = st.text_area(
                                f"Detailed Answer ({lang})", 
                                value=str(df.iloc[selected_entry].get(f"Detailed_{lang}", "")),
                                height=100,
                                key=f"edit_d_{lang}"
                            )
                    
                    col_save, col_delete = st.columns(2)
                    with col_save:
                        save_btn = st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True)
                    with col_delete:
                        delete_btn = st.form_submit_button("🗑️ Delete Entry", use_container_width=True)
                    
                    if save_btn:
                        try:
                            # Update the dataframe
                            for key, value in edited_entry.items():
                                df.at[selected_entry, key] = value
                            
                            # Save back to train.csv (or source file)
                            source_file = df.iloc[selected_entry].get('_source', 'train.csv')
                            if source_file in ['train.csv', 'test.csv'] and os.path.exists(source_file):
                                df_to_save = df[df['_source'] == source_file].drop('_source', axis=1)
                                df_to_save.to_csv(source_file, index=False)
                            else:
                                df.drop('_source', axis=1, errors='ignore').to_csv('train.csv', index=False)
                            
                            st.success("✅ Entry updated successfully!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to save changes: {e}")
                    
                    if delete_btn:
                        try:
                            # Remove entry
                            df_updated = df.drop(selected_entry).reset_index(drop=True)
                            
                            # Save to train.csv
                            df_updated.drop('_source', axis=1, errors='ignore').to_csv('train.csv', index=False)
                            
                            st.success("✅ Entry deleted successfully!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to delete entry: {e}")
    
    # TAB 4: Export Dataset
    with tab4:
        st.markdown("### 📥 Export Dataset")
        st.info(f"📊 Current dataset contains {len(df)} entries")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export as CSV")
            if st.button("📥 Download as CSV", use_container_width=True):
                csv_data = df.drop('_source', axis=1, errors='ignore').to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV File",
                    data=csv_data,
                    file_name=f"nyayasetu_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("#### Export as JSON")
            if st.button("📥 Download as JSON", use_container_width=True):
                json_data = df.drop('_source', axis=1, errors='ignore').to_json(orient='records', force_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ Download JSON File",
                    data=json_data,
                    file_name=f"nyayasetu_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        st.markdown("---")
        st.markdown("#### Preview Dataset")
        if st.checkbox("Show dataset preview"):
            st.dataframe(df.drop('_source', axis=1, errors='ignore'), use_container_width=True)

# -------------------------------
# USER MODE: Legal Consultation
# -------------------------------
else:
    st.markdown("### 💬 Get Legal Guidance Instantly")
    
    # Check if dataset is loaded
    if df.empty:
        st.warning("⚠️ No dataset loaded. Please switch to Admin Mode to upload train.csv or test.csv")
        st.stop()
    
    if not available_languages:
        st.error("❌ No valid language columns found in dataset")
        st.stop()
    
    # Example Queries
    st.markdown("**Try these example questions:**")
    example_queries = df[col_query].dropna().tolist()[:3]
    
    if example_queries:
        cols = st.columns(len(example_queries))
        for i, query in enumerate(example_queries):
            with cols[i]:
                display_text = query if len(query) <= 50 else query[:47] + "..."
                if st.button(display_text, key=f"example_{i}", use_container_width=True):
                    st.session_state.user_question = query
                    st.rerun()
    
    # User Input
    user_question = st.text_input(
        "🔍 Enter your legal question:",
        value=st.session_state.user_question,
        key='user_question_input',
        placeholder="e.g., What are my rights as a tenant?"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submitted = st.button("🔎 Get Answer", type="primary", use_container_width=True)
    with col2:
        use_rag = st.checkbox("Use RAG", value=EMBEDDINGS_AVAILABLE, disabled=not EMBEDDINGS_AVAILABLE)
    
    # Process Query
    if submitted and user_question:
        with st.spinner("🔍 Searching for your answer..."):
            queries = df[col_query].dropna().tolist()
            
            matched_row = None
            score = 0
            
            # Try RAG semantic search first
            if use_rag and EMBEDDINGS_AVAILABLE:
                results = semantic_search(user_question, queries, top_k=3)
                if results and results[0]['score'] > 0.3:
                    best_idx = results[0]['index']
                    score = results[0]['score'] * 100
                    matched_row = df.iloc[best_idx]
                    st.success(f"✅ Found relevant answer using RAG (Confidence: {score:.2f}%)")
            
            # Fallback to fuzzy matching
            if matched_row is None and FUZZY_AVAILABLE:
                try:
                    best_match, score, index = process.extractOne(
                        user_question, queries, scorer=fuzz.WRatio, score_cutoff=40
                    )
                    if best_match:
                        matched_row = df[df[col_query] == best_match].iloc[0]
                        st.success(f"✅ Found match using fuzzy search (Similarity: {score:.2f}%)")
                except Exception:
                    pass
            
            # Get answers
            if matched_row is not None:
                short_answer = str(matched_row.get(col_short, "Answer not available"))
                detailed_answer = str(matched_row.get(col_detailed, "Detailed answer not available"))
            else:
                short_answer = "❌ No relevant answer found. Please rephrase your question or contact a legal expert."
                detailed_answer = short_answer
                st.warning("No sufficiently relevant match found.")
            
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
                tts = gTTS(text=short_answer, lang=lang_code, slow=False)
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.warning(f"⚠️ Audio generation failed: {str(e)}")
            
            # Save to chat history
            st.session_state.chat_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'question': user_question,
                'answer': short_answer,
                'language': selected_lang_display,
                'confidence': score
            })
            
            # Feedback
            st.markdown("---")
            st.markdown("#### 💭 Was this helpful?")
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("👍 Yes", use_container_width=True):
                    st.success("Thanks for your positive feedback!")
            with col2:
                if st.button("👎 No", use_container_width=True):
                    st.session_state.show_feedback = True
            
            if st.session_state.get('show_feedback', False):
                feedback = st.text_area("How can we improve?", key="feedback_text")
                if st.button("Submit Feedback"):
                    st.info("✅ Feedback recorded. Thank you for helping us improve!")
                    st.session_state.show_feedback = False
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Recent Chat History")
        with st.expander("View previous questions", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                st.markdown(f"**Q{len(st.session_state.chat_history)-i}:** {chat['question']}")
                st.markdown(f"*Answer:* {chat['answer'][:150]}...")
                st.caption(f"🕐 {chat['timestamp']} | 🌐 {chat['language']} | 📊 Confidence: {chat.get('confidence', 0):.1f}%")
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

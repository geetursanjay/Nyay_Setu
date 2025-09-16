import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os
from io import BytesIO

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("SIH_Dataset_Final.xlsx")
    return df

df = load_data()

# -------------------------------
# Language mapping
# -------------------------------
language_map = {
    "English": "English",
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Marathi": "Marathi",
    "Tamil": "Tamil",
    "Telugu": "Telugu"
}

# -------------------------------
# App Title & Description
# -------------------------------
st.markdown("<h1 style='text-align:center;color:darkblue;'>⚖️ Nyayasetu - AI Legal Consultant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps. Ask your question and get structured answers based on Indian laws.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Language Selection
# -------------------------------
selected_lang = st.selectbox("Select language:", list(language_map.keys()))

col_name = f"Query_{language_map[selected_lang]}"
short_col = f"Short_{language_map[selected_lang]}"
detailed_col = f"Detailed_{language_map[selected_lang]}"

# -------------------------------
# Example Queries
# -------------------------------
if col_name in df.columns:
    example_queries = df[col_name].dropna().tolist()[:3]
    st.markdown("**Try one of these example questions:**")
    for query in example_queries:
        if st.button(query):
            st.session_state['user_question'] = query
else:
    example_queries = []

# -------------------------------
# User Input
# -------------------------------
user_question = st.text_input("Enter your question:", key='user_question')

# Allow Enter key to submit
submitted = st.button("Get Answer") or user_question.endswith("\n")

# -------------------------------
# Fetch Answer
# -------------------------------
if submitted and user_question:
    # Fuzzy match could go here; for now exact match fallback
    matched_row = df[df[col_name].str.lower() == user_question.lower()]
    
    if not matched_row.empty:
        short_answer = matched_row.iloc[0][short_col]
        detailed_answer = matched_row.iloc[0][detailed_col]
    else:
        short_answer = "❌ Sorry, we couldn't find an exact answer. Try rephrasing or select another language."
        detailed_answer = short_answer

    st.markdown(f"**Short Answer:**\n{short_answer}")
    st.markdown(f"**Detailed Answer:**\n{detailed_answer}")

    # -------------------------------
    # Text-to-Speech for English/Hindi
    # -------------------------------
    try:
        if selected_lang in ["English", "Hindi"]:
            tts = gTTS(text=short_answer, lang="hi" if selected_lang=="Hindi" else "en")
            with tempfile.NamedTemporaryFile(delete=True) as fp:
                tts.save(f"{fp.name}.mp3")
                st.audio(f"{fp.name}.mp3")
    except Exception as e:
        st.warning(f"Audio playback not available: {e}")

    # -------------------------------
    # Feedback
    # -------------------------------
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("👍", key="upvote"):
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎", key="downvote"):
            st.warning("We'll try to improve!")

# -------------------------------
# Consult a Lawyer message
# -------------------------------
st.markdown("---")
st.info("Consult a lawyer nearby → Coming soon 🚀")




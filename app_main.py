import streamlit as st
import pandas as pd
from gtts import gTTS
from io import BytesIO
from rapidfuzz import process, fuzz

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(layout="wide")

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
    .st-emotion-cache-1cypd85 {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px; padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("SIH_Dataset_Final.xlsx")
    return df

df = load_data()

# -------------------------------
# Detect available languages dynamically
# -------------------------------
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr"
}

# Check which languages exist in dataset
available_languages = []
for lang in language_map.keys():
    if f"Query_{lang}" in df.columns:
        available_languages.append(lang)

if not available_languages:
    st.error("❌ No valid language columns found in dataset. Please check your Excel file.")
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
    <p style='text-align:center; color: #000000; text-shadow: 1px 1px 2px #FFFFFF; font-weight: italic;'>Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps.</p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("Settings")
    selected_lang_display = st.selectbox("Select language:", available_languages)

# Map selected columns
col_query = f"Query_{selected_lang_display}"
col_short = f"Short_{selected_lang_display}"
col_detailed = f"Detailed_{selected_lang_display}"

# -------------------------------
# Example Queries
# -------------------------------
st.markdown("### Get Legal Guidance Instantly")
st.markdown("**Try one of these example questions:**")
example_queries = df[col_query].dropna().tolist()[:3]
cols = st.columns(len(example_queries))
for i, query in enumerate(example_queries):
    with cols[i]:
        if st.button(query, key=f"example_{i}"):
            st.session_state['user_question'] = query
            st.rerun()

# -------------------------------
# User Input
# -------------------------------
user_question = st.text_input(
    "Enter your question:",
    value=st.session_state.get('user_question', ''),
    key='user_question_input'
)

submitted = st.button("Get Answer")

# -------------------------------
# Fetch Answer
# -------------------------------
if submitted and user_question:
    with st.spinner("Searching for your answer..."):
        queries = df[col_query].dropna().tolist()

        best_match, score, index = process.extractOne(
            user_question, queries, scorer=fuzz.WRatio
        )

        if score > 80:
            matched_row = df[df[col_query] == best_match].iloc[0]
            short_answer = matched_row[col_short]
            detailed_answer = matched_row[col_detailed]
            st.success(f"✅ Found a match (Similarity: {score:.2f}%).")
        else:
            short_answer = "❌ Sorry, no close match found. Try rephrasing."
            detailed_answer = short_answer
            st.warning("No close match found.")

        # Display answers
        st.markdown(f"**Short Answer:**\n{short_answer}")
        st.markdown(f"**Detailed Answer:**\n{detailed_answer}")

        # -------------------------------
        # Text-to-Speech
        # -------------------------------
        try:
            lang_code = language_map.get(selected_lang_display, "en")
            tts = gTTS(text=short_answer, lang=lang_code)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.warning(f"Audio not available: {e}")

    # -------------------------------
    # Feedback
    # -------------------------------
    st.markdown("---")
    st.markdown("### Was this helpful?")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👍 Yes"):
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 No"):
            st.warning("We’ll improve further.")

# -------------------------------
# Coming Soon
# -------------------------------
st.markdown("---")
st.info("Consult a lawyer nearby → Coming soon 🚀")

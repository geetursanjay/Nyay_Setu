import streamlit as st
import pandas as pd
from gtts import gTTS
from io import BytesIO
from rapidfuzz import process, fuzz
from deep_translator import GoogleTranslator

# Set up page configuration for a wider layout
st.set_page_config(layout="wide")

# -------------------------------
# Custom CSS for front-end styling
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
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }
    .main-header {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        font-family: 'Dancing Script', cursive;
        color: darkblue;
        text-align: center;
    }
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px #FFFFFF;
    }
    .main-header .symbol {
        font-size: 3.5rem;
        color: #FFD700;
        text-shadow: 2px 2px 4px #000000;
    }
    .st-emotion-cache-1cypd85 {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load Dataset from GitHub
# -------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/geetursanjay/Nyay_Setu/main/SIH_Dataset_Final.xlsx"
    try:
        df = pd.read_excel(url)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

df = load_data()

# -------------------------------
# Language mapping
# -------------------------------
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te"
}

# Initialize session state
if 'user_question' not in st.session_state:
    st.session_state['user_question'] = ""

# -------------------------------
# App Title
# -------------------------------
st.markdown(
    """
    <div class="main-header">
        <span class="symbol">⚖️</span>
        <h1>Nyayasetu - AI Legal Consultant</h1>
    </div>
    <p style='text-align:center; color: #000000; text-shadow: 1px 1px 2px #FFFFFF; font-weight: italic;;'>
    Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps. 
    Ask your question and get structured answers based on Indian laws.</p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------------------
# Sidebar for Language Selection
# -------------------------------
with st.sidebar:
    st.header("Settings")
    selected_lang_display = st.selectbox("Select language:", list(language_map.keys()))
    st.info("Your feedback helps us improve!")

selected_lang_code = language_map[selected_lang_display]

# -------------------------------
# Helper: Translate text
# -------------------------------
def translate_text(text, lang_code):
    if lang_code == "en" or not text:
        return text
    try:
        return GoogleTranslator(source="en", target=lang_code).translate(text)
    except:
        return text  # fallback to English if translation fails

# -------------------------------
# Main Content Area
# -------------------------------
st.markdown("### Get Legal Guidance Instantly")
st.markdown("---")

# -------------------------------
# Example Queries
# -------------------------------
st.markdown("**Try one of these example questions:**")
example_queries = df["question"].dropna().tolist()[:3]  # always from English
example_queries_display = [
    translate_text(q, selected_lang_code) for q in example_queries
]

cols = st.columns(len(example_queries_display))
for i, query in enumerate(example_queries_display):
    with cols[i]:
        if st.button(query, key=f"example_{i}"):
            st.session_state['user_question'] = query
            st.rerun()

# -------------------------------
# User Input
# -------------------------------
input_col, button_col = st.columns([4, 1])

with input_col:
    user_question = st.text_input(
        "Enter your question:",
        value=st.session_state.get('user_question', ''),
        key='user_question_input'
    )

with button_col:
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.button("Get Answer")

# -------------------------------
# Fetch Answer
# -------------------------------
if submitted and st.session_state.get('user_question'):
    with st.spinner("Searching for your answer..."):
        queries = df["question"].dropna().tolist()

        # If user typed in regional lang, translate back to English for matching
        if selected_lang_code != "en":
            try:
                user_q_for_match = GoogleTranslator(
                    source=selected_lang_code, target="en"
                ).translate(st.session_state['user_question'])
            except:
                user_q_for_match = st.session_state['user_question']
        else:
            user_q_for_match = st.session_state['user_question']

        best_match, score, index = process.extractOne(
            user_q_for_match, queries, scorer=fuzz.WRatio
        )

        if score > 80:
            matched_row = df[df["question"] == best_match].iloc[0]
            short_answer = matched_row["answer"]
            detailed_answer = matched_row["answer"]  # if you have long form, replace here

            # Translate outputs for display
            q_display = translate_text(best_match, selected_lang_code)
            short_display = translate_text(short_answer, selected_lang_code)
            detailed_display = translate_text(detailed_answer, selected_lang_code)

            st.success(f"Found a match with a similarity score of {score:.2f}%.")

            st.markdown(f"**Q:** {q_display}")
            st.markdown(f"**Short Answer:** {short_display}")
            st.markdown(f"**Detailed Answer:** {detailed_display}")

            # -------------------------------
            # Text-to-Speech
            # -------------------------------
            try:
                if selected_lang_code:
                    tts = gTTS(text=short_display, lang=selected_lang_code)
                    audio_bytes = BytesIO()
                    tts.write_to_fp(audio_bytes)
                    audio_bytes.seek(0)
                    st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                st.warning(f"Audio playback not available: {e}")
        else:
            st.warning("❌ No close match found. Try rephrasing your query.")

    # -------------------------------
    # Feedback
    # -------------------------------
    st.markdown("---")
    st.markdown("### Was this helpful?")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("👍 Yes", key="upvote"):
            st.success("Thanks for your feedback! We'll use this to improve.")
    with col2:
        if st.button("👎 No", key="downvote"):
            st.warning("We'll try to improve the accuracy of our answers.")

# -------------------------------
# Consult a Lawyer message
# -------------------------------
st.markdown("---")
st.info("Consult a lawyer nearby → Coming soon 🚀")

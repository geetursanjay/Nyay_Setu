import streamlit as st
import pandas as pd
from gtts import gTTS
from io import BytesIO
from rapidfuzz import process, fuzz

# Set up page configuration for a wider layout
st.set_page_config(layout="wide")

# -------------------------------
# Custom CSS for front-end styling
# -------------------------------
st.markdown(
    """
    <style>
    /* Import a calligraphy font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&display=swap');

    /* Apply a background image and style */
    .stApp {
        background-image: url("https://raw.githubusercontent.com/geetursanjay/Nyay_Setu/main/background.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center center;
    }
    
    /* Add a semi-transparent overlay to the entire app */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent black overlay */
        z-index: -1;
    }
    .main-header {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px; /* Spacing between image and text */
        font-family: 'Dancing Script', cursive;
        color: darkblue;
        text-align: center;
    }
    .main-header h1 {
        font-size: 3.5rem; /* Larger font size for the title */
        font-weight: 700;
        text-shadow: 2px 2px 4px #FFFFFF; /* Changed text shadow to white for better contrast */
    }
    .main-header .symbol {
        font-size: 3.5rem; /* Increased size of the symbol to match the title */
        color: #FFD700; /* Gold color for the symbol */
        text-shadow: 2px 2px 4px #000000;
    }
    .st-emotion-cache-1cypd85 {
        background-color: rgba(255, 255, 255, 0.7); /* Lighter, semi-transparent background for content */
        border-radius: 10px;
        padding: 20px;
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
    """Loads the dataset from an Excel file and caches it."""
    try:
        df = pd.read_excel("SIH_Dataset_Final.xlsx")
        return df
    except FileNotFoundError:
        st.error("Error: 'SIH_Dataset_Final.xlsx' not found. Please ensure the file is in the same directory.")
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
# Initialize session state for the user question
if 'user_question' not in st.session_state:
    st.session_state['user_question'] = ""

# -------------------------------
# App Title with Symbol and Custom Font
# -------------------------------
st.markdown(
    """
    <div class="main-header">
        <span class="symbol">⚖️</span>
        <h1>Nyayasetu - AI Legal Consultant</h1>
    </div>
    <p style='text-align:center; color: #FFFFFF; text-shadow: 1px 1px 2px #FFFFFF;'>Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps. Ask your question and get structured answers based on Indian laws.</p>
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

# Map selected language to column names
col_name = f"Query_{selected_lang_display}"
short_col = f"Short_{selected_lang_display}"
detailed_col = f"Detailed_{selected_lang_display}"

if col_name not in df.columns or short_col not in df.columns or detailed_col not in df.columns:
    st.error(f"Error: Required columns for '{selected_lang_display}' language are not present in the dataset.")
    st.stop()
    
# -------------------------------
# Main Content Area
# -------------------------------
st.markdown("### Get Legal Guidance Instantly")
st.markdown("---")

# -------------------------------
# Example Queries
# -------------------------------
st.markdown("**Try one of these example questions:**")
example_queries = df[col_name].dropna().tolist()[:3]
cols = st.columns(len(example_queries))
for i, query in enumerate(example_queries):
    with cols[i]:
        if st.button(query, key=f"example_{i}"):
            st.session_state['user_question'] = query
            st.rerun()

# -------------------------------
# User Input
# -------------------------------
input_col, button_col = st.columns([4, 1])

with input_col:
    user_question = st.text_input("Enter your question:", value=st.session_state.get('user_question', ''), key='user_question_input')

with button_col:
    st.markdown("<br>", unsafe_allow_html=True) # Add some spacing to align the button
    submitted = st.button("Get Answer")

# -------------------------------
# Fetch Answer
# -------------------------------
if submitted and st.session_state.get('user_question'):
    with st.spinner("Searching for your answer..."):
        queries = df[col_name].dropna().tolist()
        
        best_match, score, index = process.extractOne(
            st.session_state['user_question'], 
            queries, 
            scorer=fuzz.WRatio
        )

        if score > 80:
            matched_row = df[df[col_name] == best_match]
            short_answer = matched_row.iloc[0][short_col]
            detailed_answer = matched_row.iloc[0][detailed_col]
            st.success(f"Found a match with a similarity score of {score:.2f}%.")
        else:
            short_answer = "❌ Sorry, we couldn't find a close answer. Try rephrasing or select another language."
            detailed_answer = short_answer
            st.warning("No close match found. The AI could not find a similar query in the dataset.")
            
        st.markdown(f"**Short Answer:**\n{short_answer}")
        st.markdown(f"**Detailed Answer:**\n{detailed_answer}")

        # -------------------------------
        # Text-to-Speech for all languages
        # -------------------------------
        try:
            selected_lang_code = language_map.get(selected_lang_display)
            
            if selected_lang_code:
                tts = gTTS(text=short_answer, lang=selected_lang_code)
                audio_bytes = BytesIO()
                tts.write_to_fp(audio_bytes)
                audio_bytes.seek(0)
                st.audio(audio_bytes, format='audio/mp3')
            else:
                st.warning(f"Audio playback is not supported for {selected_lang_display}.")
        except Exception as e:
            st.warning(f"Audio playback not available: {e}")

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

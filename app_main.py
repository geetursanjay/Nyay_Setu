import streamlit as st
import pandas as pd
from rapidfuzz import process
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import re

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("SIH_Dataset_Final.xlsx")
    df.columns = df.columns.str.lower()  # standardize column names
    return df

df = load_data()

# Language options
languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn"
}

# Translation function
def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception as e:
        return f"⚠️ Translation error: {e}"

# Highlight keywords in the response
def highlight_keywords(text, keywords):
    if not text:
        return text
    for kw in keywords.split():
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f"<span style='background-color:yellow; font-weight:bold;'>{kw}</span>", text)
    return text

# Search function with fuzzy matching
def get_answer(query, lang_code):
    questions = df["question"].dropna().tolist()
    best_match, score, idx = process.extractOne(query, questions, score_cutoff=40)
    if best_match:
        answer = df.iloc[idx]["answer"]
        # Translate both
        best_match_translated = translate_text(best_match, lang_code)
        answer_translated = translate_text(answer, lang_code)
        return best_match_translated, answer_translated
    return None, None

# Streamlit UI
st.title("📖 Nyay Setu – Legal Q&A Assistant")

lang_choice = st.selectbox("Choose Language", list(languages.keys()))
lang_code = languages[lang_choice]

user_query = st.text_input("Ask your legal question:")

if user_query:
    matched_q, response = get_answer(user_query, lang_code)
    if response:
        st.subheader("🔍 Closest Question:")
        st.write(matched_q)

        st.subheader("💡 Answer:")
        highlighted_answer = highlight_keywords(response, user_query)
        st.markdown(highlighted_answer, unsafe_allow_html=True)

        # Audio output
        tts = gTTS(text=response, lang=lang_code)
        tts.save("response.mp3")
        audio_file = open("response.mp3", "rb")
        st.audio(audio_file.read(), format="audio/mp3")
        audio_file.close()
        os.remove("response.mp3")
    else:
        st.warning("❌ Sorry, I couldn’t find a relevant answer. Try rephrasing your query.")

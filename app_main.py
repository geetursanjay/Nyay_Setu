#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os
from io import BytesIO
import speech_recognition as sr
from rapidfuzz import process, fuzz

# Set up page configuration for a wider layout
st.set_page_config(layout="wide")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
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
# App Title & Description
# -------------------------------
st.markdown("<h1 style='text-align:center;color:darkblue;'>⚖️ Nyayasetu - AI Legal Consultant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Nyayasetu is an AI-based legal assistant that helps citizens get quick, multi-language legal guidance in simple steps. Ask your question and get structured answers based on Indian laws.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Sidebar for Language Selection
# -------------------------------
with st.sidebar:
    st.header("Settings")
    selected_lang_display = st.selectbox("Select language:", list(language_map.keys()))
    selected_lang_code = language_map[selected_lang_display]
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
# User Input & Speech Recognition
# -------------------------------
user_input_col, speech_col = st.columns([4, 1])

with user_input_col:
    user_question = st.text_input("Enter your question:", value=st.session_state.get('user_question', ''), key='user_question_input')

with speech_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎙️ Speak"):
        st.session_state['is_listening'] = True

if st.session_state.get('is_listening'):
    r = sr.Recognizer()
    try:
        with st.spinner("Listening... Please speak clearly."):
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            
            user_question_speech = r.recognize_google(audio, language=selected_lang_code)
            st.session_state['user_question'] = user_question_speech
            st.session_state['is_listening'] = False
            st.rerun()
            
    except sr.UnknownValueError:
        st.warning("Sorry, I could not understand the audio. Please try again.")
        st.session_state['is_listening'] = False
    except sr.RequestError as e:
        st.error(f"Could not request results from the speech recognition service; {e}")
        st.session_state['is_listening'] = False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state['is_listening'] = False

# -------------------------------
# Fetch Answer
# -------------------------------
submitted = st.button("Get Answer")

if submitted and st.session_state.get('user_question'):
    with st.spinner("Searching for your answer..."):
        queries = df[col_name].dropna().tolist()
        
        # Use fuzzy matching with a threshold
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
            tts = gTTS(text=short_answer, lang=selected_lang_code)
            
            # Save the audio to a BytesIO object instead of a temporary file
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            
            st.audio(audio_bytes)
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


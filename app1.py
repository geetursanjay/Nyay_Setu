import streamlit as st
import pandas as pd
from gtts import gTTS
import tempfile
import os
import speech_recognition as sr
from rapidfuzz import process
import csv
from io import BytesIO

# Optional: AI Enhancement
try:
    from openai import OpenAI
    import os
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set in environment before running
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except:
    client = None

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

# gTTS language codes
tts_lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te"
}

# -------------------------------
# App Title & Description
# -------------------------------
st.markdown("<h1 style='text-align:center;color:darkblue;'>⚖️ Nyayasetu - AI Legal Consultant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask your legal question in your preferred language. Get quick, structured guidance powered by laws and AI.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Language Selection
# -------------------------------
selected_lang = st.selectbox("🌐 Select language:", list(language_map.keys()))

col_name = f"Query_{language_map[selected_lang]}"
short_col = f"Short_{language_map[selected_lang]}"
detailed_col = f"Detailed_{language_map[selected_lang]}"

# -------------------------------
# Example Queries
# -------------------------------
if col_name in df.columns:
    example_queries = df[col_name].dropna().tolist()[:3]
    st.markdown("💡 **Try one of these example questions:**")
    cols = st.columns(len(example_queries))
    for i, query in enumerate(example_queries):
        if cols[i].button(query):
            st.session_state['user_question'] = query
else:
    example_queries = []

# -------------------------------
# Mic Input (Speech-to-Text)
# -------------------------------
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎤 Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio, language="hi-IN" if selected_lang=="Hindi" else "en-IN")
        st.success(f"✅ You said: {text}")
        return text
    except Exception as e:
        st.error("❌ Could not recognize speech. Try again.")
        return ""

if st.button("🎤 Ask by Voice"):
    spoken_text = recognize_speech()
    if spoken_text:
        st.session_state['user_question'] = spoken_text

# -------------------------------
# User Input
# -------------------------------
user_question = st.text_input("✍️ Enter your question:", key='user_question')

# Allow Enter key to submit
submitted = st.button("🔍 Get Answer") or (user_question and user_question.endswith("\n"))

# -------------------------------
# Fetch Answer with Semantic/Fuzzy Match
# -------------------------------
if submitted and user_question:
    short_answer, detailed_answer = "", ""

    if col_name in df.columns:
        question_list = df[col_name].dropna().str.lower().tolist()
        best_match, score, idx = process.extractOne(user_question.lower(), question_list) if question_list else (None, 0, None)

        if score > 50 and idx is not None:  # threshold for fuzzy match
            matched_row = df.iloc[idx]
            short_answer = str(matched_row[short_col])
            detailed_answer = str(matched_row[detailed_col])
        else:
            short_answer = "❌ Sorry, we couldn't find a close answer in our database."
            detailed_answer = short_answer
    else:
        short_answer = "⚠️ Dataset for this language not available."
        detailed_answer = short_answer

    # -------------------------------
    # AI Enhancement (if available)
    # -------------------------------
    if client and detailed_answer not in ["", short_answer]:
        try:
            prompt = f"""You are a legal assistant. Reframe and expand the following legal answer into a clear, helpful explanation for a citizen.
            Question: {user_question}
            Existing Answer: {detailed_answer}
            """
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.5
            )
            ai_answer = response.choices[0].message.content.strip()
            detailed_answer = ai_answer
        except Exception as e:
            st.warning("⚠️ AI enhancement failed, showing dataset answer only.")

    # -------------------------------
    # Show Answers
    # -------------------------------
    st.markdown("### 📖 Answers")
    st.success(f"**Short Answer:**\n{short_answer}")
    st.info(f"**Detailed Answer:**\n{detailed_answer}")

    # -------------------------------
    # Text-to-Speech for All Languages
    # -------------------------------
    tts_lang = tts_lang_map.get(selected_lang, "en")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔈 Play Short Answer"):
            try:
                tts = gTTS(text=short_answer, lang=tts_lang)
                audio_fp = BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp.getvalue(), format="audio/mp3")
            except:
                st.warning("Audio playback failed.")
    with col2:
        if st.button("🔉 Play Detailed Answer"):
            try:
                tts = gTTS(text=detailed_answer, lang=tts_lang)
                audio_fp = BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp.getvalue(), format="audio/mp3")
            except:
                st.warning("Audio playback failed.")

    # -------------------------------
    # Feedback Section
    # -------------------------------
    st.markdown("### 📝 Feedback")
    feedback_type = st.radio("Was this answer helpful?", ["👍 Yes", "👎 No"])
    feedback_comment = ""
    if feedback_type == "👎 No":
        feedback_comment = st.text_input("Tell us what went wrong (optional):")

    if st.button("Submit Feedback"):
        file_exists = os.path.isfile("feedback.csv")
        with open("feedback.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Question", "ShortAnswer", "DetailedAnswer", "FeedbackType", "Comment"])
            writer.writerow([user_question, short_answer, detailed_answer, feedback_type, feedback_comment])
        st.success("✅ Thanks for your response! We appreciate your feedback.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.info("👨‍⚖️ Consult a lawyer nearby → Coming soon 🚀")

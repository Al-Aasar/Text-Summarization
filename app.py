import streamlit as st
from transformers import pipeline

# تحميل الموديل
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# عنوان التطبيق
st.title("📑 Text Summarization App")

# إدخال النص من المستخدم
st.subheader("Enter your text:")
user_input = st.text_area("Paste your text here", height=250)

# إعدادات التلخيص
st.sidebar.header("⚙️ Settings")
min_len = st.sidebar.slider("Minimum length", 10, 200, 30)
max_len = st.sidebar.slider("Maximum length", 50, 500, 130)

# زر التلخيص
if st.button("Summarize"):
    if user_input.strip() != "":
        with st.spinner("Summarizing... ⏳"):
            summary = summarizer(user_input, max_length=max_len, min_length=min_len, do_sample=False)
        st.subheader("📝 Summary:")
        st.write(summary[0]['summary_text'])
    else:
        st.warning("⚠️ Please enter some text to summarize.")

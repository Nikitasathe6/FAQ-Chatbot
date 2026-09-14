import streamlit as st
import json
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# NLTK RESOURCES
# ==================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ==================================================
# LOAD FAQ DATA
# ==================================================

with open("faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]


# ==================================================
# TEXT PREPROCESSING
# ==================================================

def preprocess_text(text):

    text = text.lower()

    cleaned_text = ""

    for character in text:
        if character.isalnum() or character.isspace():
            cleaned_text += character

    tokens = nltk.word_tokenize(cleaned_text)

    tokens = [
        token for token in tokens
        if len(token) > 1
    ]

    return " ".join(tokens)


# ==================================================
# PREPROCESS FAQ QUESTIONS
# ==================================================

processed_questions = [
    preprocess_text(question)
    for question in questions
]


# ==================================================
# TF-IDF MODEL
# ==================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

faq_vectors = vectorizer.fit_transform(
    processed_questions
)


# ==================================================
# FIND BEST ANSWER
# ==================================================

def get_answer(user_question):

    processed_question = preprocess_text(
        user_question
    )

    user_vector = vectorizer.transform(
        [processed_question]
    )

    if user_vector.nnz == 0:

        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please ask something related to college admission."
        )

    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarities.argmax()

    best_score = similarities[0][best_match_index]

    if best_score < 0.30:

        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please ask something related to college admission."
        )

    return answers[best_match_index]


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.10);
        text-align: center;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🤖 College FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your AI-powered college admission assistant</div>',
    unsafe_allow_html=True
)


# ==================================================
# FAQ INFORMATION
# ==================================================

st.markdown(
    f"""
    <div class="info-box">
        📚 <b>{len(faqs)} FAQs</b> available |
        💡 Ask questions about college admission
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("🤖 About Chatbot")

    st.write(
        "This chatbot uses Natural Language Processing "
        "(NLP) to find the most relevant answer from "
        "the FAQ database."
    )

    st.divider()

    st.subheader("🛠️ Technologies")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• NLTK")
    st.write("• TF-IDF")
    st.write("• Cosine Similarity")

    st.divider()

    st.subheader("💡 Example Questions")

    st.write("• How can I apply for admission?")
    st.write("• What documents are required?")
    st.write("• Can I apply online?")
    st.write("• Is hostel accommodation available?")
    st.write("• Are scholarships available?")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# CHAT HISTORY
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==================================================
# USER INPUT
# ==================================================

user_question = st.chat_input(
    "💬 Type your question here..."
)


# ==================================================
# CHATBOT RESPONSE
# ==================================================

if user_question:

    with st.chat_message("user"):

        st.write(user_question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    answer = get_answer(user_question)

    with st.chat_message("assistant"):

        st.write(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
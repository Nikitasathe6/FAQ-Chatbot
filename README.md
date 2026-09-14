# 🤖 College Admission FAQ Chatbot

An AI-powered FAQ chatbot that answers common college admission questions using Natural Language Processing (NLP), TF-IDF Vectorization, and Cosine Similarity.

This project was developed as part of my **CodeAlpha Artificial Intelligence Internship**.

---

## 📌 Project Overview

The College Admission FAQ Chatbot allows users to ask questions related to college admission and receive relevant answers instantly.

Instead of requiring users to type an exact FAQ question, the chatbot uses **TF-IDF Vectorization and Cosine Similarity** to identify the most relevant question from the FAQ database.

---

## ✨ Features

- 🤖 Interactive chatbot interface
- 🎓 College admission FAQ database
- 📚 50+ frequently asked questions
- 🧠 Natural Language Processing
- 🔤 Text preprocessing using NLTK
- 📊 TF-IDF Vectorization
- 🔍 Cosine Similarity matching
- 💬 Interactive Streamlit chat interface
- 🗑️ Clear chat functionality
- 💡 Example questions
- 🚫 Handles unrelated questions with a fallback response
- 📱 Simple and user-friendly interface

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Web application interface |
| NLTK | Natural Language Processing |
| Scikit-learn | TF-IDF and Cosine Similarity |
| JSON | FAQ database storage |
| Git & GitHub | Version control and project hosting |

---

## 🧠 How It Works

The chatbot follows these steps:

```text
User Question
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
TF-IDF Vectorization
     ↓
Cosine Similarity Matching
     ↓
Find Best Matching FAQ
     ↓
Return Answer
```

### Process Explanation

1. The user enters a college admission-related question.
2. The question is converted into lowercase text.
3. Unnecessary punctuation and unwanted characters are removed.
4. The text is tokenized using NLTK.
5. TF-IDF converts the question into numerical vectors.
6. Cosine Similarity compares the user question with all stored FAQ questions.
7. The FAQ with the highest similarity score is selected.
8. The chatbot displays the most relevant answer.
9. If no suitable match is found, a fallback response is displayed.

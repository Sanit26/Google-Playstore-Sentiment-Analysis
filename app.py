import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Page configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="📝", layout="centered")

# Custom background and styling
st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .stTextArea textarea {
            background-color: #333333;
            color: white;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
        }
        .css-18e3th9 {
            background-color: #121212;
        }
        .main {
            background-color: #121212;
            color: white;
        }
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# App title and instructions
st.title("📝 Sentiment Analysis App")
st.markdown("### 🤖 Classify your review as **Positive**, **Negative**, or **Neutral**")
st.markdown("Enter your review below and click on **Predict Sentiment** to get the result.")

# Input text
user_input = st.text_area("✍️ Enter your review here:")

# Predict button
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        # Transform and predict
        transformed = tfidf.transform([user_input])
        prediction = model.predict(transformed)[0]

        # Display sentiment
        if prediction == 'Positive':
            st.markdown(f"<h3 style='color:lightgreen;'>🙂 Sentiment: {prediction}</h3>", unsafe_allow_html=True)
        elif prediction == 'Negative':
            st.markdown(f"<h3 style='color:#FF6347;'>🙁 Sentiment: {prediction}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:lightgray;'>😐 Sentiment: {prediction}</h3>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("💡 *This app uses a machine learning model to detect sentiment from your text input.*")

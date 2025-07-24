import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load custom CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)

ssl._create_default_https_context = ssl._create_unverified_context

# Load intents
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:  
    intents = json.load(file)

# Train model
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_vector = vectorizer.transform([input_text])
    tag = clf.predict(input_vector)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

counter = 0

def main():
    global counter

    st.set_page_config(page_title="MediBot", layout="centered")
    load_css("style.css")  # ✅ Load CSS after Streamlit initializes

    st.title("🤖 MediBot")
    st.write("**Your Medicine Information Chatbot**")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.markdown("""
            Welcome to **MediBot**! I'm here to assist you with health queries about:

            - Symptoms  
            - Medications  
            - Health tips  

            Type in any question about a medicine or condition, and I’ll provide accurate responses.
        """)

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.text_area("MediBot:", value=response, height=120, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Stay healthy! 🩺")
                st.stop()

    elif choice == "Conversation History":
        st.header("🗂️ Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"🧑‍⚕️ You: {row[0]}")
                    st.text(f"🤖 MediBot: {row[1]}")
                    st.text(f"🕒 Time: {row[2]}")
                    st.markdown("---")
        else:
            st.info("No history found.")

    elif choice == "About":
        st.header("📘 About MediBot")
        st.write("""
        MediBot is a Streamlit-based chatbot designed to provide accurate and accessible medication information using Natural Language Processing and Machine Learning.

        **Key Features:**
        - Understands user queries about medicine usage, side effects, dosage, etc.
        - Trained using a custom `intents.json` file with common patterns and responses.
        - Provides a simple web UI to interact with the model.
        """)
        st.subheader("Medicines Covered")
        st.write("""
        - Paracetamol (Acetaminophen)
        - Ibuprofen
        - Metformin
        - Amlodipine
        - Atorvastatin
        - Amoxicillin
        - Pantoprazole
        - Omeprazole
        - Losartan
        - Dolo 650
        - Alerid
        - and more...
        """)
        st.warning("⚠️ This bot is for educational use only. For medical advice, consult a licensed physician.")

if __name__ == '__main__':
    main()

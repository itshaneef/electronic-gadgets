import os
import subprocess
import sys
import json
import datetime
import csv
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Check if nltk is installed, and if not, install it
try:
    import nltk
except ImportError:
    print("NLTK not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])

# Specify a custom path for NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add the custom directory to NLTK's data path
nltk.data.path.append(nltk_data_path)

# Download the 'punkt' tokenizer if not already present
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)

# This will bypass SSL verification, which is necessary for some environments
ssl._create_default_https_context = ssl._create_unverified_context

# Load intents from a JSON file (electronic gadgets dataset)
file_path = os.path.abspath("./gadgets_intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

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
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("GadgetBot")
    st.write("**Electronic Gadgets Information Chatbot**")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("""Welcome to **GadgetBot**! I'm your tech-savvy chatbot with a head full of knowledge about all things electronic gadgets. Need some quick info about your favorite devices? Just ask away!""")

        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("""**GadgetBot** is a chatbot designed to provide quick and accurate information about a wide range of electronic gadgets.""")

st.subheader("Dataset:")

st.write("""The dataset consists of a collection of labeled intents and patterns related to different electronic gadgets. Each intent has predefined patterns (user queries) and corresponding responses.""")

st.subheader("Streamlit Chatbot Interface:")

st.write("""The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their queries and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses based on user input.""")

st.subheader("Conclusion:")
st.write("""This GadgetBot is a simple chatbot designed to provide basic and relevant information about some of the most popular electronic gadgets.""")

if __name__ == '__main__':
    main()

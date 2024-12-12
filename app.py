import os
import json
import datetime
import csv
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download the necessary NLTK data package
try:
    nltk.download('punkt')
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
if not os.path.exists(file_path):
    st.error("The intents.json file was not found.")
    st.stop()

with open(file_path, "r") as file:
    intents = json.load(file)

# Train the model
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
    input_text = input_text.lower().strip()
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    return "I'm sorry, I didn't understand that."

counter = 0

def main():
    global counter
    st.title("GadgetBot")
    st.write("**Electronic Gadgets Information Chatbot**")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write(
            """
            Welcome to **GadgetBot**! I'm your friendly gadget guru. Need quick info about your gadgets? Just ask! 
            I can provide details like:
            - **Features** – What does it do?
            - **Specifications** – What are its key details?
            - **Usage Tips** – How to make the most of it?

            Please specify one of the following gadgets: Phones, Laptops, TVs, or Computers.
            """
        )

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
        if not os.path.exists('chat_log.csv'):
            st.write("No conversation history found.")
        else:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that provides accurate and reliable information about various electronic gadgets, including their features, specifications, and usage tips. The chatbot is built using Natural Language Processing (NLP) techniques to identify user intents and extract relevant entities from user queries. Streamlit, a Python web framework, is used to build an interactive web-based chatbot interface, making it easy for users to get gadget-related information quickly and efficiently.")
        st.subheader("Project Overview:")
        st.write(
            """
            Key Components:
            1. **NLP and Logistic Regression**: The chatbot uses basic logic to respond to user input and categorize it into predefined gadgets, such as Phones, Laptops, TVs, or Computers.
            2. **Streamlit Interface**: Streamlit is used to create an interactive and user-friendly interface where users can type their questions and receive responses from the chatbot.
            """
        )

        st.subheader("Dataset:")
        st.write(
            """
            The dataset consists of a collection of labeled intents and patterns related to different electronic gadgets. Each intent has predefined patterns (user queries) and corresponding responses. Here are some key intents and their examples:
            - **Greeting**: "Hi", "Hello", "Hey", "How are you"
            - **Goodbye**: "Bye", "See you later", "Goodbye", "Take care"
            - **Thanks**: "Thank you", "Thanks", "Thanks a lot", "I appreciate it"
            - **About**: "What can you do", "Who are you", "What are you", "What is your purpose"
            """
        )

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses.")

        st.subheader("Conclusion:")
        st.write(
            """
            This GadgetBot is a simple chatbot designed to provide basic and relevant information about some of the most commonly used electronic gadgets. 
            The following are the gadgets included in the bot's dataset:
            - Apple (iPhone), Samsung (Galaxy)
            - Dell (XPS), HP (Spectre)
            - Sony (Bravia), LG (OLED)
            - Lenovo (ThinkPad), Asus (ROG)

            Please note that this bot provides basic information and is not a substitute for detailed product reviews or professional tech advice.
            """
        )

if __name__ == '__main__':
    main()

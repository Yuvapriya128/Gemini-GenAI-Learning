
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Streamlit configuration
st.set_page_config(page_title="Gemini Chat Application")
st.header("Gemini LLM Application")

# Get API key
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Check your .env file.")
    st.stop()


# Initialize Gemini client only once per Streamlit session
if "gemini_client_v2" not in st.session_state:
    st.session_state.gemini_client_v2 = genai.Client(
        api_key=API_KEY
    )

client = st.session_state.gemini_client_v2

# Model configuration
MODEL = "gemini-3.5-flash-lite"


# Initialize chat session only once
if "gemini_chat_v2" not in st.session_state:
    st.session_state.gemini_chat_v2 = client.chats.create(
        model=MODEL
    )

chat = st.session_state.gemini_chat_v2


# Initialize chat history
if "chat_history_v2" not in st.session_state:
    st.session_state.chat_history_v2 = []


# Function to get streaming Gemini response
def get_gemini_response(question):

    response_stream = chat.send_message_stream(
        message=question
    )

    for chunk in response_stream:
        if chunk.text:
            yield chunk.text


# Display previous chat history
for role, message in st.session_state.chat_history_v2:

    with st.chat_message(role):
        st.write(message)


# User input
input_text = st.text_input(
    "Input:",
    key="user_question"
)

submit = st.button("Ask the question")


# Handle submitted question
if submit:

    if input_text.strip():

        # Display user question
        st.session_state.chat_history_v2.append(
            ("user", input_text)
        )

        with st.chat_message("user"):
            st.write(input_text)

        # Generate Gemini response
        with st.chat_message("assistant"):

            try:
                response = st.write_stream(
                    get_gemini_response(input_text)
                )

                # Store response in chat history
                st.session_state.chat_history_v2.append(
                    ("assistant", response)
                )

            except Exception as e:
                st.error(
                    f"Error: {type(e).__name__}: {e}"
                )

    else:
        st.warning("Please enter a question.")


import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Initialize Gemini client
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# Function to get Gemini response
def get_gemini_response(question):
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=question
    )
    return response.text


# Initialize Streamlit application
st.set_page_config(page_title="Q&A Demo")

st.header("Gemini LLM Application")

input_text = st.text_input("Input:", key="input")

submit = st.button("Ask the question")

# When submit button is clicked
if submit:
    if input_text.strip():
        with st.spinner("Generating response..."):
            try:
                response = get_gemini_response(input_text)
                st.subheader("The Response is:")
                st.write(response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")


import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize Gemini client
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Check your .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# Function to get Gemini response for an image
def get_gemini_response(prompt, image):
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[prompt, image]
    )

    return response.text


# Initialize Streamlit application
st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemini Application")

# Text prompt
input_prompt = st.text_input(
    "Input Prompt:",
    key="input"
)

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# Display uploaded image
image = None

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Unable to open image: {e}")


# Submit button
submit = st.button("Tell me about the image")

# Generate response when button is clicked
if submit:

    if image is None:
        st.warning("Please upload an image.")

    elif not input_prompt.strip():
        st.warning("Please enter a prompt.")

    else:
        try:
            with st.spinner("Gemini is analyzing the image..."):

                response = get_gemini_response(
                    input_prompt,
                    image
                )

            st.subheader("Gemini Response:")
            st.write(response)

        except Exception as e:
            st.error(f"Error generating response: {e}")

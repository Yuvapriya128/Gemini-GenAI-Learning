
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()


# ==================================================
# 2. INITIALIZE GEMINI CLIENT
# ==================================================

client = genai.Client(api_key=API_KEY)

# Model can be changed through the .env file
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


# ==================================================
# 3. GEMINI RESPONSE FUNCTION
# ==================================================

def get_gemini_response(input_prompt, image, prompt):

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            input_prompt,
            image,
            prompt
        ]
    )

    return response.text


# ==================================================
# 4. INITIALIZE STREAMLIT APPLICATION
# ==================================================

st.set_page_config(
    page_title="Multilanguage Invoice Extractor",
    page_icon="🧾",
    layout="centered"
)

st.header("Multilanguage Invoice Extractor")

st.write(
    "Upload an invoice image and extract its information "
    "using Gemini AI."
)


# ==================================================
# 5. INPUT PROMPT
# ==================================================

input_prompt = st.text_area(
    "Input Prompt:",
    value=(
        "Extract all available information from this invoice. "
        "Include invoice number, invoice date, due date, "
        "seller details, buyer details, item descriptions, "
        "quantities, unit prices, subtotal, tax, discounts, "
        "total amount, currency and payment details. "
        "Preserve the original values and do not invent "
        "missing information."
    ),
    height=130,
    key="input"
)


# ==================================================
# 6. IMAGE UPLOADER
# ==================================================

uploaded_file = st.file_uploader(
    "Choose an image of the invoice...",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Invoice",
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Unable to open the uploaded image: {e}"
        )


# ==================================================
# 7. ADDITIONAL EXTRACTION INSTRUCTIONS
# ==================================================

prompt = """
You are an expert invoice information extraction assistant.

Analyze the uploaded invoice image carefully.

Requirements:

1. Extract all visible invoice information.
2. Support invoices written in different languages.
3. Preserve the original text, names, dates and amounts.
4. Do not guess or fabricate missing information.
5. If a field is missing or unreadable, mention "Not available".
6. Preserve the original currency and numerical values.
7. Identify the invoice language if possible.
8. Organize the extracted information using clear headings.
9. Present line items in a readable table.
10. Mention any unclear or illegible information separately.

Return the extracted invoice information in a structured,
easy-to-read format.
"""


# ==================================================
# 8. SUBMIT BUTTON
# ==================================================

submit = st.button(
    "Extract Invoice Details",
    type="primary"
)


# ==================================================
# 9. GENERATE GEMINI RESPONSE
# ==================================================

if submit:

    if image is None:

        st.warning("Please upload an invoice image first.")

    elif not input_prompt.strip():

        st.warning("Please enter an extraction prompt.")

    else:

        try:

            with st.spinner(
                "Gemini is analyzing your invoice..."
            ):

                response = get_gemini_response(
                    input_prompt,
                    image,
                    prompt
                )

            if response:

                st.subheader("Extracted Invoice Details")

                st.markdown(response)

            else:

                st.warning(
                    "Gemini returned an empty response. "
                    "Please try again."
                )

        except Exception as e:

            st.error(
                f"Error: {type(e).__name__}: {e}"
            )

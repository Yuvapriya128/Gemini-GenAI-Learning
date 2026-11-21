
import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()


# ==================================================
# 2. STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Chat With Multiple PDFs",
    page_icon="📚"
)

st.header("Chat with Multiple PDF Documents")


# ==================================================
# 3. INITIALIZE GEMINI MODELS
# ==================================================

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-001"

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=API_KEY
)

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    temperature=0.3,
    google_api_key=API_KEY
)


# ==================================================
# 4. EXTRACT TEXT FROM PDF FILES
# ==================================================

def get_pdf_text(pdf_docs):

    documents = []

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page_number, page in enumerate(
            pdf_reader.pages,
            start=1
        ):

            text = page.extract_text() or ""

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf.name,
                            "page": page_number
                        }
                    )
                )

    return documents


# ==================================================
# 5. SPLIT TEXT INTO CHUNKS
# ==================================================

def get_text_chunks(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# ==================================================
# 6. CREATE FAISS VECTOR STORE
# ==================================================

def get_vector_store(text_chunks):

    vector_store = FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")

    return vector_store


# ==================================================
# 7. CREATE CONVERSATIONAL QA CHAIN
# ==================================================

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible using
    the provided context.

    If the answer is not available in the context,
    clearly say "Answer is not available in the context".

    Do not provide information outside the given context.

    Context:
    {context}

    Previous conversation:
    {chat_history}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "chat_history",
            "question"
        ]
    )

    return prompt | llm


# ==================================================
# 8. INITIALIZE SESSION STATE
# ==================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==================================================
# 9. HANDLE USER QUESTIONS
# ==================================================

def user_input(user_question):

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process PDF files first."
        )

        return

    try:

        with st.spinner(
            "Searching PDF documents and generating answer..."
        ):

            # Retrieve relevant PDF chunks
            relevant_documents = (
                st.session_state.vector_store.similarity_search(
                    user_question,
                    k=4
                )
            )

            # Prepare context
            context = "\n\n".join(
                document.page_content
                for document in relevant_documents
            )

            # Prepare recent chat history
            chat_history = "\n".join(
                f"{role}: {message}"
                for role, message in st.session_state.chat_history[-6:]
            )

            # Generate answer
            conversation_chain = get_conversational_chain()

            response = conversation_chain.invoke(
                {
                    "context": context,
                    "chat_history": chat_history,
                    "question": user_question
                }
            )

            answer = response.content

        # Display answer
        st.subheader("Reply:")
        st.write(answer)

        # Store conversation
        st.session_state.chat_history.append(
            ("User", user_question)
        )

        st.session_state.chat_history.append(
            ("Assistant", answer)
        )

        # Display source information
        with st.expander("View PDF Sources"):

            for document in relevant_documents:

                st.write(
                    f"**File:** {document.metadata.get('source')}"
                )

                st.write(
                    f"**Page:** {document.metadata.get('page')}"
                )

                st.write(
                    document.page_content[:500]
                )

                st.divider()

    except Exception as e:

        st.error(
            f"Error: {type(e).__name__}: {e}"
        )


# ==================================================
# 10. MAIN STREAMLIT APPLICATION
# ==================================================

user_question = st.text_input(
    "Ask a Question from the PDF Files"
)

if st.button("Submit Question"):

    if user_question.strip():

        user_input(user_question)

    else:

        st.warning("Please enter a question.")


# ==================================================
# 11. SIDEBAR - UPLOAD AND PROCESS PDF FILES
# ==================================================

with st.sidebar:

    st.title("Menu")

    pdf_docs = st.file_uploader(
        "Upload your PDF Files and Click on Submit & Process",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Submit & Process"):

        if not pdf_docs:

            st.warning("Please upload PDF files.")

        else:

            try:

                with st.spinner("Processing..."):

                    # Extract PDF text
                    raw_documents = get_pdf_text(pdf_docs)

                    if not raw_documents:

                        st.warning(
                            "No readable text found in the uploaded PDFs."
                        )

                        st.stop()

                    # Split extracted text
                    text_chunks = get_text_chunks(
                        raw_documents
                    )

                    # Create vector store
                    vector_store = get_vector_store(
                        text_chunks
                    )

                    st.session_state.vector_store = vector_store

                    # Reset chat history for new PDFs
                    st.session_state.chat_history = []

                st.success("Done!")

                st.write(
                    f"PDF files processed: {len(pdf_docs)}"
                )

                st.write(
                    f"Text chunks created: {len(text_chunks)}"
                )

            except Exception as e:

                st.error(
                    f"PDF processing error: {type(e).__name__}: {e}"
                )


# ==================================================
# 12. DISPLAY CHAT HISTORY
# ==================================================

if st.session_state.chat_history:

    st.subheader("Chat History")

    for role, message in st.session_state.chat_history:

        with st.chat_message(
            "user" if role == "User" else "assistant"
        ):

            st.write(message)

import io
import gc
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

# Title and description (no emojis)
st.title("My Document Question Answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get "
    "[here](https://platform.openai.com/account/api-keys)."
)

# API key
openai_api_key = st.text_input("OpenAI API key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
else:
    # Create client once (no global persistence beyond this run)
    client = OpenAI(api_key=openai_api_key)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .md, or .pdf)",
        type=("txt", "md", "pdf")
    )

    # Question input (disabled until a file is present)
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=uploaded_file is None,
    )

    # Only process when both file and question are present
    if uploaded_file and question:
        document = None
        file_bytes = None

        # Get extension defensively
        name = uploaded_file.name or ""
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

        # Read content in-memory without persisting
        if ext in ("txt", "md"):
            document = uploaded_file.read().decode("utf-8", errors="replace")

        elif ext == "pdf":
            file_bytes = uploaded_file.read()
            reader = PdfReader(io.BytesIO(file_bytes))
            text_pages = []
            for page in reader.pages:
                text_pages.append(page.extract_text() or "")
            document = "\n".join(text_pages).strip()
        else:
            st.error("Unsupported file type.")
            st.stop()

        if not document:
            st.warning("Could not extract any text from the uploaded file.")
            st.stop()

        # Build messages and call the API
        messages = [
            {"role": "user",
             "content": f"Here's a document: {document}\n\n---\n\n{question}"}
        ]

        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)

        # Explicitly clear sensitive variables after use
        try:
            del document
        except NameError:
            pass
        try:
            del messages
        except NameError:
            pass
        if file_bytes is not None:
            file_bytes = b""
        gc.collect()

    # If the user removes the file, there is no retained copy:
    # - We never store file contents in st.session_state or on disk.
    # - Variables are recreated each run and cleared after use.


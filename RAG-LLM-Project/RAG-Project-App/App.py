import streamlit as st
from PyPDF2 import PdfReader
from fpdf import FPDF
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


st.title("(LangChain + Gemini)")

# Upload PDF @st.cache
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
pdf_text = ""
if uploaded_file:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        temp_pdf_path = tmp_pdf.name
    # Extract text for preview
    pdf_reader = PdfReader(temp_pdf_path)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() or ""
    st.subheader("PDF Preview")
    st.write(pdf_text[:1500])

# ========== BLOCK 2: Gemini API Key Input ==========
api_key = st.text_input("Enter your Gemini API Key:", type="password")

all_chunks = []
vectordb = None

if uploaded_file and api_key:
    # Load PDF with LangChain
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    # Save all chunks for reference
    all_chunks = [doc.page_content for doc in docs]
    # Create vector DB using MiniLM embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_store")
    st.success("PDF processed")

# ========== BLOCK 4: User Question ==========
user_question = st.text_input("Ask a question about your PDF:")

# ========== BLOCK 5: RAG Retrieval ==========
retrieved_context = ""
if vectordb and user_question:
    # Get similar chunks (context)
    retriever = vectordb.as_retriever(search_kwargs={"k":3})
    relevant_docs = retriever.get_relevant_documents(user_question)
    # Join for prompt
    context_chunks = []
    for doc in relevant_docs:
        context_chunks.append(doc.page_content)
    retrieved_context = "\n---\n".join(context_chunks)
    st.subheader("Retrieved Contextfor Gemini API")
    st.write(retrieved_context[:1000])

# ========== BLOCK 6: Gemini Prompt & Response ==========
answer = ""
if api_key and retrieved_context and user_question:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash-001")
        prompt = f"""Context:
{retrieved_context}

Question: {user_question}
Answer:"""
        resp = gemini_model.generate_content(prompt)
        answer = resp.text
        st.subheader("Gemini's Resp")
        st.write(answer)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")

# ========== BLOCK 7: Add Answer to PDF ==========
if answer and uploaded_file:
    # Create new PDF with original pages + answer page
    pdf_reader = PdfReader(temp_pdf_path)
    new_pdf = FPDF()
    # Add original pages
    for i in range(len(pdf_reader.pages)):
        new_pdf.add_page()
        page_text = pdf_reader.pages[i].extract_text() or ""
        for chunk in [page_text[j:j+800] for j in range(0, len(page_text), 800)]:
            new_pdf.set_font("Arial", size=12)
            new_pdf.multi_cell(0, 10, chunk)
    # Add answer as last page
    new_pdf.add_page()
    new_pdf.set_font("Arial", 'B', 14)
    new_pdf.cell(0, 10, "Gemini Generated Answer", ln=True)
    new_pdf.set_font("Arial", size=12)
    # Easy for loop to add lines
    for line in answer.split('\n'):
        new_pdf.multi_cell(0, 10, line)
    # Save to temp file
    out_path = os.path.join(tempfile.gettempdir(), "pdf_with_answer.pdf")
    new_pdf.output(out_path)
    # Download button
    with open(out_path, "rb") as f:
        st.download_button("Download PDF with Answer", f, file_name="pdf_with_answer.pdf")

if uploaded_file:
    os.remove(temp_pdf_path)
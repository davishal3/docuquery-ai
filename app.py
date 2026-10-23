# Streamlit ties all these modules together. We use st.session_state to ensure the PDF is only processed once, preventing it from re-embedding every time the user types a question.
import streamlit as st
import os
from dotenv import load_dotenv

from ingestion import extract_text_from_pdf, chunk_text
from embeddings import get_embeddings, get_query_embedding
from retrieval import DocumentVectorStore
from generation import generate_grounded_answer

load_dotenv()

st.set_page_config(page_title="DocuQuery AI", layout="centered")

st.title("📄 DocuQuery AI")
st.markdown("Ask questions. Get answers grounded in your documents.")

# Check for API key
if not os.getenv("GEMINI_API_KEY"):
    st.error("Please set your GEMINI_API_KEY in the .env file.")
    st.stop()

# Initialize session state for our vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# Sidebar for file upload
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and not st.session_state.doc_processed:
        with st.spinner("Processing document..."):
            # 1. Extract text
            pages = extract_text_from_pdf(uploaded_file)
            
            # 2. Chunk text
            chunks = chunk_text(pages, chunk_size=400, overlap=50)
            
            # 3. Embed chunks
            texts_to_embed = [c["text"] for c in chunks]
            embeddings = get_embeddings(texts_to_embed)
            
            # 4. Store in FAISS
            vector_store = DocumentVectorStore(embedding_dimension=384)
            vector_store.add_chunks(chunks, embeddings)
            
            st.session_state.vector_store = vector_store
            st.session_state.doc_processed = True
            
        st.success(f"Processed {len(pages)} pages into {len(chunks)} chunks.")

# Main chat interface
st.header("2. Ask Questions")
if st.session_state.doc_processed:
    query = st.text_input("What would you like to know about this document?")
    
    if query:
        with st.spinner("Searching and generating answer..."):
            # 1. Embed query
            query_emb = get_query_embedding(query)
            
            # 2. Retrieve top passages
            retrieved_chunks = st.session_state.vector_store.search(query_emb, top_k=3)
            
            # 3. Generate Answer
            answer = generate_grounded_answer(query, retrieved_chunks)
            
            # Display Answer
            st.markdown("### Answer")
            st.write(answer)
            
            # Level 2 Feature: Show retrieved sources for verification
            with st.expander("View Retrieved Passages"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {i+1} (Page {chunk['page_number']})**")
                    st.write(chunk['text'])
                    st.divider()
else:
    st.info("Please upload a PDF in the sidebar to begin.")
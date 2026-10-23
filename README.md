# docuquery-ai
A lightweight RAG document assistant built with Python, Streamlit, FAISS, and the Gemini API. Upload PDFs, ask natural language questions, and get grounded answers backed by exact page citations. Features a modular architecture and built-in evaluation for retrieval precision and latency.
# 📄 DocuQuery AI — RAG-Based Document Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green)
![Gemini API](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-orange)

**DocuQuery AI** is a lightweight, fully functional Retrieval-Augmented Generation (RAG) application. It allows users to upload PDF documents, ask natural language questions, and receive precise answers grounded directly in the document's contents, complete with page number citations. 

Built to demonstrate core GenAI engineering principles—document chunking, local vector embeddings, semantic search, strict prompt grounding, and automated evaluation—without the overhead of heavy orchestration frameworks.

---

## 🚀 Key Features

* **Intelligent Document Ingestion:** Extracts text page-by-page while preserving critical metadata (page numbers) using `pypdf`.
* **Optimized Chunking:** Implements an overlapping sliding window text-splitter (400 tokens, 50 token overlap) to maintain contextual boundaries across paragraphs.
* **Local Semantic Search:** Utilizes `sentence-transformers` (`all-MiniLM-L6-v2`) for generating 384-dimensional dense vectors, indexed locally with **FAISS** for millisecond-latency retrieval.
* **Grounded LLM Generation:** Connects directly to the **Google Gemini 2.5 Flash REST API** (bypassing SDK dependency issues), utilizing strict system prompting to enforce source-grounding and prevent hallucinations.
* **Source Transparency:** The UI surfaces the exact retrieved passages and page numbers used to formulate the answer, allowing for immediate user verification.
* **Built-in Evaluation:** Includes a custom evaluation script to measure **Retrieval Precision@k**, pipeline latency, and the LLM's abstention behavior (its ability to decline unanswerable questions).

---

## 🛠️ Technology Stack

* **Frontend:** Streamlit
* **Document Processing:** PyPDF
* **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Vector Store:** FAISS (CPU)
* **LLM Engine:** Google Gemini 2.5 Flash (via Direct REST API)
* **Environment Management:** Python `python-dotenv`

---

## 📂 Project Structure

```text
docuquery-ai/
├── app.py                 # Streamlit UI and application state
├── ingestion.py           # PDF extraction and overlapping chunking logic
├── embeddings.py          # Sentence-transformer embedding generation
├── retrieval.py           # FAISS index creation and similarity search
├── generation.py          # LLM prompt engineering and REST API integration
├── evaluation.py          # Script for testing precision and latency
├── requirements.txt       # Project dependencies
├── .env.example           # Template for environment variables
└── README.md              # Project documentation
## Local Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/davishal3/docuquery-ai.git
cd docuquery-ai
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your Google Gemini API key:

```ini
GEMINI_API_KEY=your_actual_api_key_here
```

## 💻 Usage

### 1. Run the Streamlit Interface

To start the chat interface, run the following command in your activated virtual environment:

```bash
streamlit run app.py
```

Upload a PDF via the sidebar, wait a few seconds for the document to be chunked and indexed, and begin asking questions.

### 2. Run the Evaluation Pipeline

To test the retrieval accuracy and response latency of the system against a ground-truth dataset:

1. Place a sample PDF in the project directory.
2. Update the `GROUND_TRUTH_TESTS` dictionary in `evaluation.py` with questions and expected page numbers specific to your document.
3. Run the evaluation script:

```bash
python evaluation.py
```

The terminal will output the Retrieval Precision@3 score, average latency, and demonstrate whether the LLM successfully abstains from answering out-of-context questions.

---

**Author:** Vishal Malik | [GitHub](https://github.com/davishal3)

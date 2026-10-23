import time
from ingestion import extract_text_from_pdf, chunk_text
from embeddings import get_embeddings, get_query_embedding
from retrieval import DocumentVectorStore
from generation import generate_grounded_answer

def run_evaluation(pdf_path, test_cases):
    print(f"--- Starting Evaluation on {pdf_path} ---")
    
    # 1. Ingestion Pipeline
    print("Processing document...")
    start_time = time.time()
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages, chunk_size=400, overlap=50)
    
    texts_to_embed = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts_to_embed)
    
    vector_store = DocumentVectorStore(embedding_dimension=384)
    vector_store.add_chunks(chunks, embeddings)
    
    ingestion_time = time.time() - start_time
    print(f"Document processed in {ingestion_time:.2f} seconds.\n")

    # 2. Evaluation Variables
    total_questions = len(test_cases)
    successful_retrievals = 0
    total_latency = 0

    # 3. Run Test Cases
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['question']}")
        
        q_start_time = time.time()
        
        # Retrieval
        query_emb = get_query_embedding(test['question'])
        retrieved_chunks = vector_store.search(query_emb, top_k=3)
        
        # Check Precision (Did we retrieve the expected page?)
        retrieved_pages = [chunk['page_number'] for chunk in retrieved_chunks]
        if test['expected_page'] in retrieved_pages:
            successful_retrievals += 1
            print(f"✅ Retrieval Success (Found on expected page {test['expected_page']})")
        else:
            print(f"❌ Retrieval Failed (Expected {test['expected_page']}, got {retrieved_pages})")
            
        # Generation
        answer = generate_grounded_answer(test['question'], retrieved_chunks)
        
        q_latency = time.time() - q_start_time
        total_latency += q_latency
        
        print(f"Answer: {answer[:150]}...\n")
        print(f"Latency: {q_latency:.2f} seconds\n")
        print("-" * 50)

    # 4. Final Metrics
    precision = (successful_retrievals / total_questions) * 100
    avg_latency = total_latency / total_questions
    
    print("\n=== FINAL EVALUATION METRICS ===")
    print(f"Retrieval Precision@3: {precision:.1f}%")
    print(f"Average Answer Latency: {avg_latency:.2f} seconds")


if __name__ == "__main__":
    # To run this, place a sample PDF in your project folder
    # and update the PDF_FILE path below.
    PDF_FILE = "sample.pdf" # Replace with your actual PDF filename
    
    # Create a small ground-truth dataset based on your specific PDF.
    # Update these questions and expected pages based on what is actually in your PDF.
    GROUND_TRUTH_TESTS = [
        {
            "question": "What is the main topic of this document?",
            "expected_page": 1 
        },
        {
            "question": "What methodology or process is described?",
            "expected_page": 2
        },
        {
            "question": "What is the capital of France?", # Unanswerable question test
            "expected_page": -1 # -1 because it shouldn't be in the document
        }
    ]
    
    try:
        run_evaluation(PDF_FILE, GROUND_TRUTH_TESTS)
    except FileNotFoundError:
        print(f"Error: Could not find '{PDF_FILE}'. Please put a PDF in the folder and update the PDF_FILE variable.")
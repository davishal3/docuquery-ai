import os
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_grounded_answer(query, retrieved_chunks):
    # Get API key directly from .env
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is missing in .env file."

    # Format the retrieved chunks into a single context string
    context_text = ""
    for idx, chunk in enumerate(retrieved_chunks):
        context_text += f"--- Chunk {idx+1} (Page {chunk['page_number']}) ---\n"
        context_text += chunk['text'] + "\n\n"

    prompt = f"""You are DocuQuery AI, a strict document-grounded question-answering assistant.

RULES:
1. Answer the user's question using ONLY the supplied Document Context.
2. Cite the exact page number(s) for every claim you make.
3. If the context does not contain enough information, say clearly: "The uploaded document does not contain sufficient evidence to answer this question."
4. Do not invent facts.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{query}
"""

    # URL updated to Gemini 2.5 Flash (1.5 is deprecated and deleted by Google)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        # Send direct HTTP request
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        
        # Check if Google returned an error
        if response.status_code != 200:
            return f"Google API Error: {response_json.get('error', {}).get('message', 'Unknown error')}"
            
        # Parse the answer
        answer = response_json['candidates'][0]['content']['parts'][0]['text']
        return answer
        
    except Exception as e:
        return f"System Error: {str(e)}"
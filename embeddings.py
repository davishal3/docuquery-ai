# We'll use a small, efficient local model (all-MiniLM-L6-v2) to turn our text chunks into numerical vectors.from sentence_transformers import SentenceTransformer

# Load the model once when the module is imported
# This is a small, fast model generating 384-dimensional vectors
from sentence_transformers import SentenceTransformer

# Load the model once when the module is imported
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    """
    Takes a list of strings and returns their embeddings as a NumPy array.
    """
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings

def get_query_embedding(query):
    """
    Generates an embedding for a single user query.
    """
    return model.encode([query])[0]
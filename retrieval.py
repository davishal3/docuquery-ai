# This file manages FAISS. It stores our vectors and metadata (the actual text and page numbers) 
# so we can retrieve them later.

import faiss
import numpy as np

class DocumentVectorStore:
    def __init__(self, embedding_dimension=384):
        # L2 distance (Euclidean) is standard for FAISS flat indexes
        self.index = faiss.IndexFlatL2(embedding_dimension)
        self.metadata = []

    def add_chunks(self, chunks, embeddings):
        """Adds text chunks and their vectors to the store."""
        # FAISS requires float32 numpy arrays
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.metadata.extend(chunks)

    def search(self, query_embedding, top_k=3):
        """Searches the index for the most similar vectors."""
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, top_k)
        
        results = []
        # indices[0] contains the indices of the nearest neighbors
        for i in indices[0]:
            if i != -1 and i < len(self.metadata):
                results.append(self.metadata[i])
                
        return results
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class DocumentQA:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = self.client.create_collection(
            name="qa_documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add_document(self, text, metadata=None):
        """Add a document to the vector store"""
        # Split document into chunks for better retrieval
        chunks = self._chunk_text(text, chunk_size=500)

        for i, chunk in enumerate(chunks):
            embedding = self.model.encode(chunk).tolist()
            doc_id = f"doc_{len(self.collection.get()['ids'])}_{i}"

            # Prepare the add parameters
            add_params = {
                "embeddings": embedding,
                "documents": [chunk],
                "ids": [doc_id]
            }
            
            # Only add metadata if it's provided and non-empty
            if metadata:
                add_params["metadatas"] = [metadata]
            
            self.collection.add(**add_params)

    def _chunk_text(self, text, chunk_size=500):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def answer_question(self, question, n_results=3):
        """Answer a question using retrieved documents"""
        # Retrieve relevant documents
        query_embedding = self.model.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        # Combine Retrieved documents
        context = "\n\n".join(results['documents'][0])

        # Generate answer using OpenAI with new client syntax
        response = self.openai_client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. If you cannot answer based on the context, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            max_tokens=200,
            temperature=0.1
        )

        return response.choices[0].message.content, results['documents'][0]

# Example usage
qa_system = DocumentQA()

# Add Samples
documents = [
    "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They use indexing techniques like HNSW (Hierarchical Navigable Small World) to enable fast similarity search",
    "Machine learning models convert data into numerical representations called embeddings. These embeddings capture semantic meaning and can be used for various tasks like search, recommendation, and classification",
    "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge sources. It retrieves relevant information from a knowledge base and uses it to generate more accurate and contextual responses"
]

for doc in documents:
    qa_system.add_document(doc)

# Ask questions
question = "What are vector databases used for?"
answer, sources = qa_system.answer_question(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"\nSources Used:")
for i, source in enumerate(sources, 1):
    print(f"{i}. {source[:100]}...")
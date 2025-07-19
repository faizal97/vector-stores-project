# Project 2: Document Q&A System - Complete Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [What is RAG (Retrieval-Augmented Generation)?](#what-is-rag-retrieval-augmented-generation)
3. [System Architecture](#system-architecture)
4. [Technical Prerequisites](#technical-prerequisites)
5. [Dependencies and Imports](#dependencies-and-imports)
6. [DocumentQA Class - Line-by-Line Analysis](#documentqa-class---line-by-line-analysis)
7. [Mathematical Concepts](#mathematical-concepts)
8. [Vector Database Fundamentals](#vector-database-fundamentals)
9. [Embedding Models Deep Dive](#embedding-models-deep-dive)
10. [Example Usage Analysis](#example-usage-analysis)
11. [Performance Considerations](#performance-considerations)
12. [Troubleshooting](#troubleshooting)
13. [Next Steps and Improvements](#next-steps-and-improvements)

---

## Overview

This project implements a **Document Question & Answer (Q&A) System** using **Retrieval-Augmented Generation (RAG)**. The system allows users to:

1. Store documents in a vector database
2. Ask natural language questions about those documents
3. Receive accurate answers based on the stored content

The implementation combines three key technologies:
- **Vector Databases** (ChromaDB) for efficient document storage and retrieval
- **Sentence Transformers** for converting text into numerical embeddings
- **Large Language Models** (OpenAI GPT) for generating human-like responses

---

## What is RAG (Retrieval-Augmented Generation)?

### The Problem RAG Solves

Traditional Large Language Models (LLMs) like GPT have two main limitations:
1. **Knowledge Cutoff**: They only know information from their training data
2. **Hallucination**: They sometimes generate plausible-sounding but incorrect information

### How RAG Works

RAG solves these problems by combining two steps:

1. **Retrieval Phase**: 
   - Convert the user's question into a vector (numerical representation)
   - Search a knowledge base for the most relevant documents
   - Retrieve the top matching documents

2. **Generation Phase**:
   - Provide the retrieved documents as context to an LLM
   - Generate an answer based on this specific context
   - Ensure the answer is grounded in actual source material

### Mathematical Foundation

The RAG process can be mathematically represented as:

```
P(answer|question) = Σ P(answer|question, document_i) × P(document_i|question)
```

Where:
- `P(answer|question)` is the probability of generating a specific answer
- `P(document_i|question)` is the relevance score of document i to the question
- `P(answer|question, document_i)` is the probability of generating the answer given both question and document

---

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Text    │───▶│  Sentence        │───▶│   ChromaDB      │
│   Documents     │    │  Transformer     │    │  Vector Store   │
└─────────────────┘    │  (Embedding)     │    └─────────────────┘
                       └──────────────────┘             │
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│   User          │───▶│  Query           │────────────┘
│   Question      │    │  Processing      │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐
│   Generated     │◀───│   OpenAI GPT     │
│   Answer        │    │   LLM            │
└─────────────────┘    └──────────────────┘
```

---

## Technical Prerequisites

### Required Concepts

Before diving into the code, you should understand:

1. **Vectors and Embeddings**: Numerical representations of text that capture semantic meaning
2. **Cosine Similarity**: A method to measure how similar two vectors are
3. **API Keys**: Authentication tokens for external services
4. **Object-Oriented Programming**: Classes, methods, and instantiation

### Mathematical Background

**Vector Similarity**: Given two vectors A and B, cosine similarity is calculated as:
```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

Where:
- `A · B` is the dot product
- `|A|` and `|B|` are the magnitudes of the vectors

---

## Dependencies and Imports

Let's analyze each import statement in detail:

### Line 1: `import chromadb`
```python
import chromadb
```

**ChromaDB** is an open-source vector database specifically designed for AI applications. Key features:
- **Purpose**: Stores high-dimensional vectors (embeddings) efficiently
- **Indexing**: Uses advanced algorithms like HNSW for fast similarity search
- **Persistence**: Can store data permanently or in-memory
- **Similarity Search**: Finds the most similar vectors to a query vector

**Why We Need It**: Traditional databases store structured data (numbers, strings). Vector databases store embeddings and can quickly find semantically similar content.

### Line 2: `from sentence_transformers import SentenceTransformer`
```python
from sentence_transformers import SentenceTransformer
```

**Sentence Transformers** is a library that converts text into dense vector representations:
- **Pre-trained Models**: Uses models trained on massive text datasets
- **Semantic Understanding**: Captures meaning, not just keywords
- **Dense Vectors**: Typically 384 or 768 dimensions
- **Multilingual Support**: Many models support multiple languages

**Technical Detail**: The model we use (`all-MiniLM-L6-v2`) creates 384-dimensional vectors.

### Line 3: `from openai import OpenAI`
```python
from openai import OpenAI
```

**OpenAI API Client** provides access to GPT models:
- **Language Generation**: Creates human-like text responses
- **Context Awareness**: Can understand and respond to provided context
- **Fine-tuning**: Can adjust responses based on instructions
- **Multiple Models**: Access to different GPT variants

### Lines 4-5: Environment and Configuration
```python
import os
from dotenv import load_dotenv
```

**Environment Management**:
- `os`: Provides access to operating system functionality
- `dotenv`: Loads environment variables from a `.env` file
- **Security**: Keeps API keys separate from source code

### Line 7: `load_dotenv()`
```python
load_dotenv()
```

**Function**: Searches for a `.env` file and loads its variables into the environment.
**Security Benefit**: API keys are stored separately from code, preventing accidental exposure.

---

## DocumentQA Class - Line-by-Line Analysis

### Class Definition and Constructor

#### Lines 9-10: Class Declaration
```python
class DocumentQA:
    def __init__(self):
```

**Object-Oriented Design**: Creates a class that encapsulates all Q&A functionality.
**Constructor**: The `__init__` method runs when creating a new instance.

#### Line 11: Embedding Model Initialization
```python
self.model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Technical Breakdown**:
- **Model Name**: `all-MiniLM-L6-v2` is a specific pre-trained model
- **Architecture**: Based on MiniLM, optimized for sentence embeddings
- **Dimensions**: Produces 384-dimensional vectors
- **Performance**: Balance between speed and accuracy
- **Size**: Approximately 90MB model

**Mathematical Process**: The model uses transformer architecture with attention mechanisms to convert text into vectors that capture semantic meaning.

#### Line 12: Vector Database Client
```python
self.client = chromadb.Client()
```

**In-Memory Database**: Creates a ChromaDB client that stores data temporarily.
**Production Note**: For persistent storage, you would use `chromadb.PersistentClient(path="./data")`.

#### Line 13: OpenAI Client Setup
```python
self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**API Authentication**: 
- Retrieves API key from environment variables
- Creates authenticated client for OpenAI services
- **Security**: Never hardcode API keys in source code

#### Lines 14-17: Collection Creation
```python
self.collection = self.client.create_collection(
    name="qa_documents",
    metadata={"hnsw:space": "cosine"}
)
```

**Collection Details**:
- **Name**: `qa_documents` - identifier for this specific collection
- **Metadata**: Configuration parameters for the vector index
- **HNSW**: Hierarchical Navigable Small World - an efficient algorithm for approximate nearest neighbor search
- **Cosine Space**: Uses cosine similarity for distance calculations

**HNSW Algorithm**: Creates a multi-layer graph structure that enables logarithmic search time complexity O(log n) instead of linear O(n).

### Document Addition Method

#### Lines 19-22: Method Signature and Documentation
```python
def add_document(self, text, metadata=None):
    """Add a document to the vector store"""
    # Split document into chunks for better retrieval
    chunks = self._chunk_text(text, chunk_size=500)
```

**Chunking Strategy**:
- **Problem**: Long documents may lose important details when converted to a single embedding
- **Solution**: Split documents into smaller, overlapping chunks
- **Chunk Size**: 500 words balances context preservation with specificity
- **Benefit**: Improves retrieval accuracy for specific questions

#### Lines 24-26: Chunk Processing Loop
```python
for i, chunk in enumerate(chunks):
    embedding = self.model.encode(chunk).tolist()
    doc_id = f"doc_{len(self.collection.get()['ids'])}_{i}"
```

**Line-by-Line Breakdown**:
- `enumerate(chunks)`: Provides both index `i` and content `chunk`
- `self.model.encode(chunk)`: Converts text chunk to 384-dimensional vector
- `.tolist()`: Converts numpy array to Python list (required by ChromaDB)
- **ID Generation**: Creates unique identifier using collection size and chunk index

**Mathematical Process**: The `encode` method applies multiple transformer layers with attention mechanisms to create a dense vector representation.

#### Lines 28-33: Parameter Preparation
```python
# Prepare the add parameters
add_params = {
    "embeddings": embedding,
    "documents": [chunk],
    "ids": [doc_id]
}
```

**Data Structure**: Prepares parameters for database insertion:
- **embeddings**: The 384-dimensional vector
- **documents**: Original text content
- **ids**: Unique identifier for retrieval

#### Lines 35-39: Conditional Metadata Addition
```python
# Only add metadata if it's provided and non-empty
if metadata:
    add_params["metadatas"] = [metadata]

self.collection.add(**add_params)
```

**Metadata Handling**:
- **Optional**: Only adds metadata if provided
- **Use Cases**: Document source, creation date, author, etc.
- **Dictionary Unpacking**: `**add_params` expands dictionary as keyword arguments

### Text Chunking Method

#### Lines 41-50: Chunking Implementation
```python
def _chunk_text(self, text, chunk_size=500):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
```

**Algorithm Analysis**:
- **Word-Based Splitting**: More natural than character-based splitting
- **Window Size**: 500 words per chunk
- **No Overlap**: This implementation doesn't overlap chunks (could be improved)
- **Range Function**: `range(0, len(words), chunk_size)` creates non-overlapping windows

**Potential Improvement**: Adding overlap (e.g., 50-word overlap) would prevent losing context at chunk boundaries.

### Question Answering Method

#### Lines 52-55: Method Setup
```python
def answer_question(self, question, n_results=3):
    """Answer a question using retrieved documents"""
    # Retrieve relevant documents
    query_embedding = self.model.encode(question).tolist()
```

**Query Processing**:
- **Same Model**: Uses identical model for consistency
- **Vector Representation**: Question becomes 384-dimensional vector
- **Semantic Search**: Will find documents with similar meaning, not just keywords

#### Lines 56-59: Vector Search
```python
results = self.collection.query(
    query_embeddings=query_embedding,
    n_results=n_results
)
```

**Database Query**:
- **Similarity Search**: Finds most similar vectors to the question
- **n_results=3**: Returns top 3 most relevant chunks
- **Algorithm**: Uses HNSW for efficient approximate nearest neighbor search

**Mathematical Process**: Calculates cosine similarity between query vector and all stored vectors, returns highest scoring matches.

#### Line 62: Context Assembly
```python
context = "\n\n".join(results['documents'][0])
```

**Context Preparation**:
- **Concatenation**: Combines retrieved chunks into single context
- **Separator**: Uses double newline for clear separation
- **First Result Set**: `[0]` accesses the first (and only) query's results

#### Lines 65-73: LLM Response Generation
```python
response = self.openai_client.chat.completions.create(
    model="gpt-4.1-nano-2025-04-14",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context. If you cannot answer based on the context, say so."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ],
    max_tokens=200,
    temperature=0.1
)
```

**API Parameters Explained**:
- **model**: Specific GPT model version
- **messages**: Conversation format with system and user messages
- **System Message**: Instructions for how the AI should behave
- **User Message**: Combines retrieved context with user question
- **max_tokens=200**: Limits response length
- **temperature=0.1**: Low temperature for more deterministic, factual responses

**Temperature Scale**: 
- 0.0 = Completely deterministic
- 1.0 = Maximum creativity/randomness
- 0.1 = Slight randomness while maintaining factual accuracy

#### Line 75: Return Results
```python
return response.choices[0].message.content, results['documents'][0]
```

**Return Value**: Tuple containing:
1. **Generated Answer**: The LLM's response text
2. **Source Documents**: The original chunks used for context

---

## Mathematical Concepts

### Vector Embeddings

**Definition**: Embeddings are dense vector representations of text where similar meanings are close in vector space.

**Example**: 
- "car" might be represented as [0.2, -0.1, 0.8, ...]
- "automobile" might be [0.3, -0.05, 0.7, ...] (similar values)
- "banana" might be [-0.1, 0.9, -0.3, ...] (different values)

### Similarity Calculation

**Cosine Similarity Formula**:
```
similarity = (A · B) / (|A| × |B|)
```

**Example Calculation**:
```python
import numpy as np

# Two example vectors
vector_a = np.array([1, 2, 3])
vector_b = np.array([2, 3, 4])

# Dot product
dot_product = np.dot(vector_a, vector_b)  # 20

# Magnitudes
magnitude_a = np.linalg.norm(vector_a)  # 3.74
magnitude_b = np.linalg.norm(vector_b)  # 5.39

# Cosine similarity
similarity = dot_product / (magnitude_a * magnitude_b)  # 0.99
```

### HNSW Algorithm Complexity

**Search Complexity**: O(log n) where n is the number of vectors
**Space Complexity**: O(n × M) where M is the maximum number of connections per node

---

## Vector Database Fundamentals

### Why Vector Databases?

**Traditional Databases**:
- Store exact data (numbers, strings, dates)
- Use exact matching or simple comparisons
- Example: `SELECT * WHERE name = 'John'`

**Vector Databases**:
- Store high-dimensional numerical vectors
- Use similarity search instead of exact matching
- Example: Find documents similar to "machine learning"

### ChromaDB Architecture

**Components**:
1. **Collections**: Logical groupings of documents
2. **Embeddings**: Vector representations of documents
3. **Metadata**: Additional information about documents
4. **Indices**: Data structures for fast search (HNSW)

**Storage Options**:
- **In-Memory**: Fast but temporary (`chromadb.Client()`)
- **Persistent**: Slower but permanent (`chromadb.PersistentClient()`)

---

## Embedding Models Deep Dive

### all-MiniLM-L6-v2 Specifications

**Architecture**: 
- Based on Microsoft's MiniLM
- 6 transformer layers
- 384 hidden dimensions
- 22.7 million parameters

**Training Data**:
- 1 billion sentence pairs
- Multiple languages
- Diverse domains (web, books, news)

**Performance Metrics**:
- **Speed**: ~500 sentences/second on CPU
- **Accuracy**: 82.37% on STS benchmark
- **Size**: 90MB model file

### Alternative Models

**Larger Models** (better accuracy, slower):
- `all-mpnet-base-v2`: 768 dimensions, 420MB
- `all-roberta-large-v1`: 1024 dimensions, 1.3GB

**Smaller Models** (faster, lower accuracy):
- `all-MiniLM-L12-v2`: 384 dimensions, 90MB
- `paraphrase-MiniLM-L3-v2`: 384 dimensions, 60MB

---

## Example Usage Analysis

### Lines 78-88: System Initialization and Document Loading
```python
qa_system = DocumentQA()

# Add Samples
documents = [
    "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They use indexing techniques like HNSW (Hierarchical Navigable Small World) to enable fast similarity search",
    "Machine learning models convert data into numerical representations called embeddings. These embeddings capture semantic meaning and can be used for various tasks like search, recommendation, and classification",
    "Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge sources. It retrieves relevant information from a knowledge base and uses it to generate more accurate and contextual responses"
]

for doc in documents:
    qa_system.add_document(doc)
```

**Process Flow**:
1. **Instantiation**: Creates all necessary components (models, database, API client)
2. **Document Processing**: Each document is:
   - Split into chunks (these are short, so likely 1 chunk each)
   - Converted to embeddings using SentenceTransformer
   - Stored in ChromaDB with unique IDs

### Lines 91-98: Question Processing and Output
```python
question = "What are vector databases used for?"
answer, sources = qa_system.answer_question(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"\nSources Used:")
for i, source in enumerate(sources, 1):
    print(f"{i}. {source[:100]}...")
```

**Execution Steps**:
1. **Question Embedding**: "What are vector databases used for?" → 384-dimensional vector
2. **Similarity Search**: Find most similar document chunks
3. **Context Assembly**: Combine retrieved chunks
4. **Answer Generation**: GPT processes context and question
5. **Result Display**: Shows answer and source attribution

**Expected Retrieval**: The first document about vector databases should rank highest due to semantic similarity.

---

## Performance Considerations

### Computational Complexity

**Embedding Generation**:
- **Time**: O(sequence_length) for transformer processing
- **Memory**: O(batch_size × sequence_length × hidden_size)

**Vector Search**:
- **HNSW Time**: O(log n) average case
- **Memory**: O(n × dimensions) for vector storage

**LLM Generation**:
- **Time**: O(output_tokens) - sequential generation
- **Memory**: O(context_length) for attention computation

### Optimization Strategies

#### 1. Batch Processing
```python
# Instead of processing one document at a time
embeddings = model.encode(chunks)  # Batch processing
```

#### 2. Persistent Storage
```python
# For production use
client = chromadb.PersistentClient(path="./vector_db")
```

#### 3. Caching
```python
# Cache frequently accessed embeddings
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_encode(text):
    return model.encode(text)
```

### Memory Usage

**Typical Memory Requirements**:
- **SentenceTransformer Model**: ~200MB RAM
- **ChromaDB Collection**: ~4 bytes × dimensions × documents
- **OpenAI API**: Minimal local memory (cloud-based)

**Example Calculation**:
- 10,000 documents × 384 dimensions × 4 bytes = ~15MB for vectors
- Plus metadata and text storage

---

## Troubleshooting

### Common Issues and Solutions

#### 1. OpenAI API Key Errors
**Error**: `openai.AuthenticationError: Incorrect API key`

**Solutions**:
- Verify `.env` file exists with `OPENAI_API_KEY=your_key_here`
- Check API key validity on OpenAI dashboard
- Ensure no extra spaces or quotes in the key

#### 2. Model Download Issues
**Error**: `OSError: Can't load tokenizer for 'all-MiniLM-L6-v2'`

**Solutions**:
```bash
# Clear cache and reinstall
pip uninstall sentence-transformers
pip install sentence-transformers
```

#### 3. ChromaDB Collection Errors
**Error**: `Collection qa_documents already exists`

**Solutions**:
```python
# Delete existing collection
try:
    client.delete_collection("qa_documents")
except:
    pass
collection = client.create_collection("qa_documents")
```

#### 4. Memory Issues with Large Documents
**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce chunk size: `chunk_size=200`
- Process documents in smaller batches
- Use CPU instead of GPU: `device='cpu'`

#### 5. Poor Answer Quality
**Symptoms**: Irrelevant or incorrect answers

**Solutions**:
- Increase `n_results` parameter: `n_results=5`
- Improve chunking with overlap
- Filter low-similarity results
- Add more relevant documents to the knowledge base

### Debug Techniques

#### 1. Inspect Embeddings
```python
# Check embedding dimensions and values
embedding = model.encode("test text")
print(f"Shape: {embedding.shape}")
print(f"Sample values: {embedding[:5]}")
```

#### 2. Analyze Similarity Scores
```python
# Get similarity scores for debugging
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,
    include=['documents', 'distances']
)
print(f"Similarity scores: {results['distances']}")
```

#### 3. Trace Retrieval Process
```python
# Log each step
print(f"Query: {question}")
print(f"Retrieved docs: {len(results['documents'][0])}")
print(f"Context length: {len(context)} characters")
```

---

## Next Steps and Improvements

### Short-term Enhancements

#### 1. Improved Chunking Strategy
```python
def _chunk_text_with_overlap(self, text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        end_idx = min(i + chunk_size, len(words))
        chunk = " ".join(words[i:end_idx])
        chunks.append(chunk)
        
        if end_idx == len(words):
            break
    
    return chunks
```

#### 2. Similarity Threshold Filtering
```python
def answer_question(self, question, n_results=3, min_similarity=0.7):
    # ... existing code ...
    
    # Filter by similarity threshold
    filtered_docs = []
    for doc, distance in zip(results['documents'][0], results['distances'][0]):
        similarity = 1 - distance  # Convert distance to similarity
        if similarity >= min_similarity:
            filtered_docs.append(doc)
    
    if not filtered_docs:
        return "I don't have enough relevant information to answer this question.", []
```

#### 3. Document Metadata Enhancement
```python
qa_system.add_document(
    text=document_text,
    metadata={
        "source": "research_paper.pdf",
        "date": "2024-01-15",
        "author": "Dr. Smith",
        "topic": "machine_learning"
    }
)
```

### Medium-term Improvements

#### 1. Multi-Modal Support
- Add support for images and PDFs
- Implement OCR for scanned documents
- Support for structured data (tables, charts)

#### 2. Advanced Retrieval Strategies
- **Hybrid Search**: Combine vector similarity with keyword matching
- **Re-ranking**: Use cross-encoders for better relevance scoring
- **Query Expansion**: Automatically expand queries with synonyms

#### 3. Persistent Storage with Backup
```python
class DocumentQA:
    def __init__(self, persist_directory="./vector_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # ... rest of initialization
        
    def backup_database(self, backup_path):
        """Create backup of vector database"""
        import shutil
        shutil.copytree(self.persist_directory, backup_path)
```

### Long-term Architecture Evolution

#### 1. Microservices Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web API       │───▶│  Embedding       │───▶│   Vector DB     │
│   Gateway       │    │  Service         │    │   Service       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│   Question      │    │   Document       │
│   Service       │    │   Processing     │
└─────────────────┘    │   Service        │
                       └──────────────────┘
```

#### 2. Real-time Updates
- Implement document change detection
- Add incremental indexing
- Support for document versioning

#### 3. Advanced Analytics
- Track query patterns and popular questions
- Monitor answer quality metrics
- A/B testing for different retrieval strategies

### Performance Scaling

#### 1. Production Deployment Considerations
```python
# Production-ready configuration
class ProductionDocumentQA(DocumentQA):
    def __init__(self):
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Production ChromaDB settings
        self.client = chromadb.HttpClient(
            host="your-chroma-server.com",
            port=8000
        )
        
        # Connection pooling for OpenAI
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=3,
            timeout=30
        )
```

#### 2. Monitoring and Logging
```python
import logging
import time

class MonitoredDocumentQA(DocumentQA):
    def answer_question(self, question, n_results=3):
        start_time = time.time()
        
        try:
            result = super().answer_question(question, n_results)
            processing_time = time.time() - start_time
            
            logging.info(f"Question answered in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logging.error(f"Error answering question: {e}")
            raise
```

---

## Conclusion

This Document Q&A System demonstrates the power of combining vector databases, embedding models, and large language models to create an intelligent information retrieval system. The implementation provides a solid foundation that can be extended and scaled for production use.

Key learning points:
- **RAG Architecture**: Understanding how retrieval and generation work together
- **Vector Embeddings**: How text becomes numerical representations
- **Similarity Search**: Efficient algorithms for finding relevant information
- **LLM Integration**: Using context to generate accurate, grounded responses

The system showcases modern AI techniques that are fundamental to many applications including chatbots, search engines, and knowledge management systems.

---

## File Location
This README.md file is located at: `/Users/fayz/Documents/Work/Projects/Personal/vector-stores-project/project2/README.md`

Main implementation file: `/Users/fayz/Documents/Work/Projects/Personal/vector-stores-project/project2/main.py`
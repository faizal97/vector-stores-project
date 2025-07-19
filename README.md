# Vector Stores Project

A collection of Python implementations demonstrating different use cases for vector databases and similarity search using ChromaDB.

## Projects Overview

### Project 1: Basic Document Search
Simple semantic search implementation using sentence transformers and ChromaDB.

**Features:**
- Document embedding using SentenceTransformer
- Cosine similarity search
- Sample dataset with diverse content

**Key Dependencies:**
- `chromadb`
- `sentence-transformers`

### Project 2: Document Q&A System
Retrieval-Augmented Generation (RAG) system combining vector search with OpenAI GPT for question answering.

**Features:**
- Document chunking for better retrieval
- Vector-based document retrieval
- GPT-powered answer generation
- Context-aware responses

**Key Dependencies:**
- `chromadb`
- `sentence-transformers`
- `openai`
- `python-dotenv`

### Project 3: Image Similarity Search
Multi-modal search system using CLIP for image and text embeddings.

**Features:**
- Image-to-image similarity search
- Text-to-image search
- CLIP model integration
- Support for local images and URLs

**Key Dependencies:**
- `chromadb`
- `torch`
- `transformers`
- `Pillow`

### Project 4: Product Recommendation System
Content-based recommendation system using TF-IDF vectorization.

**Features:**
- Product embedding from descriptions
- User profile creation from purchase history
- Similarity-based recommendations
- Purchase history filtering

**Key Dependencies:**
- `chromadb`
- `scikit-learn`
- `pandas`
- `numpy`

## Setup

1. Install required dependencies:
```bash
pip install chromadb sentence-transformers openai python-dotenv torch transformers Pillow scikit-learn pandas numpy
```

2. For Project 2, create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. For Project 3, ensure you have the sample images in the project3 directory:
- `cat.jpg`
- `dog.jpg` 
- `car.jpg`
- `tree.jpg`

## Usage

Navigate to each project directory and run the main.py file:

```bash
cd project1
python main.py
```

Each project demonstrates different aspects of vector databases and similarity search, from basic text search to advanced multi-modal applications.
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB Client
client = chromadb.Client()

# Create a collection
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Python is powerful programming language",
    "Machine Learning algorithms can predicts outcomes",
    "Natural language processing helps computer understand text",
    "Deep learning models require large amounts of data",
    "Vector databases store high-dimensional data efficiently",
    "Semantic search finds meaning beyond keywords",
    "Thrall meets jaina in the khaz algar",
    "Varian wrynn attacks Zuljin",
    "The Alliance need help from The Horde",
    "Cristiano Ronaldo is the best player in the world",
    "The Witcher 3: Wild Hunt is a fantasy RPG game",
    "Sylvanas Windrunner becomes the new Lich Queen",
    "Arthas Menethil falls to the corruption of Frostmourne",
    "The Burning Legion invades Azeroth once again",
    "Illidan Stormrage sacrifices everything for his people",
    "Basketball is one of the most popular sports worldwide",
    "The Theory of Relativity was proposed by Einstein",
    "Mount Everest is Earth's highest mountain above sea level",
    "The Renaissance period began in Italy in the 14th century",
    "Space exploration continues to advance with new technologies",
    "Deathwing's emergence reshapes the world of Azeroth",
    "Climate change affects global weather patterns",
    "The Internet revolutionized global communication"
]

# Generate embeddings and add to collection
for i, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()
    collection.add(
        embeddings=embedding,
        documents=[doc],
        ids=[str(uuid.uuid4())]
    )

# Search function
def search_documents(query, n_results=3):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results['documents'][0]

# Test the search
query = "Warcraft"
results = search_documents(query)
print(f"Query: {query}")
print("Results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result}")
import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class RecommendationSystem:
    def __init__(self):
        self.client = chromadb.Client()
        # We'll create collections after knowing embedding dimensions
        self.products_collection = None
        self.users_collection = None
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
        self.products_data = []  # Store products until we create embeddings

    def add_product(self, product_id, name, description, category, price):
        """Add a product to the recommendation system"""
        # Create product embedding from description
        product_text = f"{name} {description} {category}"

        # Store product data temporarily until we create embeddings
        self.products_data.append({
            "id": product_id,
            "text": product_text,
            "name": name,
            "description": description,
            "category": category,
            "price": price
        })

    def _update_embeddings(self):
        """Update all product embeddings after all products are added"""
        if not self.products_data:
            return

        # Extract all text documents
        documents = [product["text"] for product in self.products_data]
        
        # Fit TF-IDF on all documents  
        self.tfidf.fit(documents)
        
        # Generate embeddings for all products
        embeddings = self.tfidf.transform(documents).toarray()
        
        
        # Create collections with correct dimensions
        embedding_dim = embeddings.shape[1]
        
        try:
            self.client.delete_collection("products")
        except:
            pass
        try:
            self.client.delete_collection("users")
        except:
            pass
            
        self.products_collection = self.client.create_collection(
            name="products",
            metadata={"hnsw:space": "cosine"}
        )
        self.users_collection = self.client.create_collection(
            name="users", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add products with embeddings
        ids = [product["id"] for product in self.products_data]
        metadatas = [{
            "name": product["name"],
            "description": product["description"], 
            "category": product["category"],
            "price": product["price"]
        } for product in self.products_data]
        
        self.products_collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    def create_user_profile(self, user_id, purchased_products, preferences=None):
        """Create user profile based on purchase history"""
        # Get embeddings of purchased products
        if not purchased_products:
            return

        product_embeddings = []
        for product_id in purchased_products:
            try:
                result = self.products_collection.get(ids=[product_id], include=['embeddings'])
                if result['embeddings'] is not None and len(result['embeddings']) > 0:
                    product_embeddings.append(result['embeddings'][0])
            except Exception as e:
                continue

        if not product_embeddings:
            return

        # Create user profile as average of purchased products
        user_embedding = np.mean(product_embeddings, axis=0).tolist()

        try:
            self.users_collection.add(
                embeddings=[user_embedding],
                documents=[f"User {user_id} profile"],
                ids=[user_id],
                metadatas=[{
                    "purchased_products": ",".join(purchased_products),
                    "preferences": str(preferences or {})
                }]
            )
        except Exception as e:
            pass

    def recommend_products(self, user_id, n_recommendations=5, exclude_purchased=True):
        """Recommend products to a user"""
        # Get user profile
        user_result = self.users_collection.get(ids=[user_id], include=['embeddings', 'metadatas'])
        if user_result['embeddings'] is None or len(user_result['embeddings']) == 0:
            return []

        user_embedding = user_result['embeddings'][0]
        purchased_products = user_result['metadatas'][0].get('purchased_products', "").split(",") if user_result['metadatas'][0].get('purchased_products') else []

        # find similar products
        try:
            results = self.products_collection.query(
                query_embeddings=[user_embedding],
                n_results=n_recommendations + len(purchased_products)
            )
        except Exception as e:
            return []

        recommendations = []
        
        for i, (product_id, metadata) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
            if exclude_purchased and product_id in purchased_products:
                continue

            distance = results['distances'][0][i]
            similarity = 1 - distance
            
            recommendations.append({
                "product_id": product_id,
                "name": metadata["name"],
                "description": metadata["description"],
                "category": metadata["category"],
                "price": metadata["price"],
                "similarity_score": similarity
            })

            if (len(recommendations) >= n_recommendations):
                break

        return recommendations

# Example usage
rec_system = RecommendationSystem()

# Add sample products with more descriptive features
products = [
    ("prod1", "iPhone 13", "Latest premium smartphone with advanced camera technology high-quality device portable", "Electronics", 999),
    ("prod2", "MacBook Pro", "High-performance premium laptop for professionals quality device portable technology", "Electronics", 1999),
    ("prod3", "Running Shoes", "Comfortable premium athletic shoes for running high-quality fitness equipment", "Sports", 129),
    ("prod4", "Yoga Mat", "Premium quality yoga mat for exercise fitness equipment", "Sports", 49),
    ("prod5", "Coffee Maker", "Automatic premium coffee brewing machine high-quality home appliance", "Home", 199)
]

for product in products:
    rec_system.add_product(*product)

# Update embeddings after adding all products
rec_system._update_embeddings()

# Create user profile
user_purchases = ["prod1", "prod2"]
rec_system.create_user_profile("user123", user_purchases)

# Get recommendations
recommendations = rec_system.recommend_products("user123", n_recommendations=3)

print("Recommendations for user123:")
for rec in recommendations:
    print(f"- {rec['name']} (${rec['price']}) - Score: {rec['similarity_score']:.3f}")

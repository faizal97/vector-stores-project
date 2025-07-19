# Project 4: Product Recommendation System

## Table of Contents
1. [Introduction to Recommendation Systems](#introduction-to-recommendation-systems)
2. [Understanding TF-IDF](#understanding-tf-idf)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Code Architecture Overview](#code-architecture-overview)
5. [Detailed Code Explanation](#detailed-code-explanation)
6. [System Components](#system-components)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps and Improvements](#next-steps-and-improvements)

## Introduction to Recommendation Systems

A **recommendation system** is an intelligent algorithm that suggests items to users based on their preferences, behavior, or characteristics. These systems are everywhere: Netflix recommends movies, Amazon suggests products, and Spotify recommends music.

### Types of Recommendation Systems

1. **Content-Based Filtering**: Recommends items similar to those a user has previously liked
2. **Collaborative Filtering**: Recommends items based on similar users' preferences
3. **Hybrid Approach**: Combines multiple techniques

This project implements a **content-based filtering** system using TF-IDF vectorization and cosine similarity to recommend products based on textual descriptions.

## Understanding TF-IDF

**TF-IDF** (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document within a collection of documents.

### Components

1. **Term Frequency (TF)**: How often a term appears in a document
2. **Inverse Document Frequency (IDF)**: How rare or common a term is across all documents

### Why TF-IDF Works for Recommendations

- It identifies unique characteristics of products
- Common words (like "the", "and") get lower weights
- Distinctive product features get higher weights
- Creates meaningful numerical representations of text

## Mathematical Foundations

### TF-IDF Formula

```
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
```

Where:
- `t` = term (word)
- `d` = document (product description)
- `D` = collection of all documents

### Term Frequency (TF)
```
TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)
```

### Inverse Document Frequency (IDF)
```
IDF(t,D) = log(Total number of documents / Number of documents containing term t)
```

### Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- Values range from -1 to 1
- 1 = identical vectors
- 0 = orthogonal vectors
- -1 = opposite vectors

## Code Architecture Overview

The system consists of three main components:

1. **Product Storage**: ChromaDB vector database for storing product embeddings
2. **User Profiles**: Mathematical representation of user preferences
3. **Recommendation Engine**: Similarity calculation and ranking system

## Detailed Code Explanation

### Imports and Dependencies

```python
import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
```

**Line-by-line explanation:**
- **Line 1**: `chromadb` - A vector database for storing and querying high-dimensional embeddings efficiently
- **Line 2**: `numpy` - Essential for numerical computations and array operations
- **Line 3**: `TfidfVectorizer` - Scikit-learn's implementation of TF-IDF text vectorization
- **Line 4**: `cosine_similarity` - Function to calculate cosine similarity between vectors
- **Line 5**: `pandas` - Data manipulation library (imported but not used in this implementation)

### RecommendationSystem Class Definition

```python
class RecommendationSystem:
    def __init__(self):
        self.client = chromadb.Client()
        # We'll create collections after knowing embedding dimensions
        self.products_collection = None
        self.users_collection = None
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
        self.products_data = []  # Store products until we create embeddings
```

**Line-by-line explanation:**

**Line 7**: Class definition for our recommendation system
**Line 8**: Constructor method that initializes the system
**Line 9**: Creates a ChromaDB client instance for vector database operations
**Line 11**: Placeholder for products collection (initialized as None)
**Line 12**: Placeholder for users collection (initialized as None)
**Line 13**: TfidfVectorizer configuration:
- `stop_words="english"`: Removes common English words ("the", "and", "is", etc.)
- `ngram_range=(1,2)`: Creates both unigrams (single words) and bigrams (two-word phrases)
- `min_df=1`: Include terms that appear in at least 1 document
**Line 14**: Temporary storage for product data before creating embeddings

### Adding Products to the System

```python
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
```

**Line-by-line explanation:**

**Line 16**: Method definition for adding products
**Line 19**: **Text Concatenation**: Combines product name, description, and category into a single text string. This creates a comprehensive textual representation that captures all important product features.

**Why concatenate these fields?**
- Name: Contains brand and product type information
- Description: Contains detailed features and characteristics
- Category: Provides context and grouping information

**Lines 22-29**: **Data Structure Creation**: Creates a dictionary containing all product information. This temporary storage allows us to collect all products before generating embeddings, which is more efficient than processing them one by one.

### Updating Embeddings

```python
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
```

**Line-by-line explanation:**

**Line 31**: Private method (indicated by underscore prefix) for updating embeddings
**Line 33-34**: **Guard Clause**: Returns early if no products exist
**Line 37**: **Document Extraction**: Creates a list of all product text descriptions using list comprehension
**Line 40**: **TF-IDF Fitting**: 
- Analyzes all documents to build vocabulary
- Calculates IDF values for each term
- Creates the mathematical model for text vectorization

**Line 43**: **Embedding Generation**:
- `transform()`: Converts text documents to TF-IDF vectors
- `.toarray()`: Converts sparse matrix to dense NumPy array for compatibility

### Database Collection Setup

```python
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
```

**Line-by-line explanation:**

**Line 47**: **Dimension Calculation**: Gets the number of features (dimensions) in the TF-IDF vectors
**Lines 49-56**: **Collection Cleanup**: 
- Attempts to delete existing collections to avoid conflicts
- Uses try-except blocks because deletion fails if collections don't exist

**Lines 58-65**: **Collection Creation**:
- Creates separate collections for products and users
- `hnsw:space": "cosine"`: Configures ChromaDB to use cosine distance for similarity calculations
- HNSW (Hierarchical Navigable Small World) is an efficient algorithm for approximate nearest neighbor search

### Adding Products to Vector Database

```python
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
```

**Line-by-line explanation:**

**Line 68**: **ID Extraction**: Creates list of product IDs
**Lines 69-74**: **Metadata Creation**: Creates list of dictionaries containing product information for each item
**Lines 76-81**: **Database Insertion**:
- `embeddings`: TF-IDF vectors converted to lists
- `documents`: Original text descriptions
- `ids`: Unique identifiers for each product
- `metadatas`: Additional product information for retrieval

### Creating User Profiles

```python
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
```

**Line-by-line explanation:**

**Line 83**: Method for creating user preference profiles
**Line 86-87**: **Empty Purchase Check**: Returns if user has no purchase history
**Line 89**: **Embedding Storage**: List to collect product embeddings
**Lines 90-96**: **Embedding Retrieval Loop**:
- Iterates through each purchased product
- Retrieves the TF-IDF embedding from the database
- Handles errors gracefully (e.g., if product doesn't exist)
- Only includes valid embeddings

### User Profile Calculation

```python
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
```

**Line-by-line explanation:**

**Lines 98-99**: **Validation Check**: Ensures we have embeddings to work with
**Line 102**: **Profile Generation**: 
- `np.mean(product_embeddings, axis=0)`: Calculates the average vector across all purchased products
- This creates a user profile representing their aggregate preferences
- `axis=0`: Averages along the first dimension (across products, keeping feature dimensions)

**Lines 104-115**: **User Profile Storage**:
- Stores the user profile as an embedding in the users collection
- Includes metadata about purchased products and preferences
- Uses error handling for database operations

### Product Recommendation Engine

```python
def recommend_products(self, user_id, n_recommendations=5, exclude_purchased=True):
    """Recommend products to a user"""
    # Get user profile
    user_result = self.users_collection.get(ids=[user_id], include=['embeddings', 'metadatas'])
    if user_result['embeddings'] is None or len(user_result['embeddings']) == 0:
        return []

    user_embedding = user_result['embeddings'][0]
    purchased_products = user_result['metadatas'][0].get('purchased_products', "").split(",") if user_result['metadatas'][0].get('purchased_products') else []
```

**Line-by-line explanation:**

**Line 117**: Method definition with parameters:
- `user_id`: Identifier for the user
- `n_recommendations`: Number of recommendations to return (default 5)
- `exclude_purchased`: Whether to exclude already purchased items (default True)

**Line 120**: **User Profile Retrieval**: Gets user embedding and metadata from database
**Lines 121-122**: **Profile Validation**: Returns empty list if user profile doesn't exist
**Line 124**: **Embedding Extraction**: Gets the user's preference vector
**Line 125**: **Purchase History**: Extracts list of previously purchased products from metadata

### Similarity Search and Ranking

```python
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
```

**Line-by-line explanation:**

**Lines 128-134**: **Vector Search**:
- `query_embeddings`: User's preference vector for similarity search
- `n_results`: Requests extra results to account for filtering purchased items
- ChromaDB returns the most similar products based on cosine similarity

**Line 136**: **Recommendation List**: Storage for final recommendations

**Lines 138-156**: **Result Processing Loop**:
- **Line 138**: Iterates through search results with enumeration for indexing
- **Lines 139-140**: **Purchase Filter**: Skips products the user already owns
- **Line 142**: **Distance Extraction**: Gets the cosine distance from search results
- **Line 143**: **Similarity Conversion**: Converts distance to similarity score (1 - distance)
- **Lines 145-152**: **Recommendation Object**: Creates structured recommendation with all relevant information
- **Lines 154-155**: **Limit Check**: Stops when we have enough recommendations

### Example Usage and Testing

```python
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
```

**Line-by-line explanation:**

**Line 160**: **System Instantiation**: Creates a new recommendation system instance

**Lines 163-169**: **Sample Data**: Creates test products with enhanced descriptions:
- Keywords like "premium", "high-quality" are repeated to increase TF-IDF weights
- Descriptions include multiple relevant terms for better similarity matching

**Lines 171-172**: **Product Addition**: Adds all products using tuple unpacking (`*product`)

**Line 175**: **Embedding Generation**: Processes all products to create TF-IDF vectors

**Lines 177-179**: **User Profile Creation**: Creates a profile for a user who purchased iPhone and MacBook (both electronics)

**Lines 181-186**: **Recommendation Generation and Display**: 
- Gets 3 recommendations for the user
- Displays results with similarity scores formatted to 3 decimal places

## System Components

### 1. Vector Database (ChromaDB)
- **Purpose**: Efficient storage and retrieval of high-dimensional embeddings
- **Features**: HNSW indexing for fast similarity search
- **Collections**: Separate storage for products and users

### 2. TF-IDF Vectorizer
- **Configuration**: 
  - English stop words removal
  - Unigrams and bigrams (1-2 word phrases)
  - Minimum document frequency of 1
- **Output**: Sparse vectors representing text importance

### 3. User Profile System
- **Method**: Centroid-based approach (average of purchased product embeddings)
- **Benefits**: Captures aggregate user preferences
- **Limitations**: Equal weight to all purchases (no recency or frequency weighting)

### 4. Recommendation Algorithm
- **Approach**: Content-based filtering using cosine similarity
- **Ranking**: Products sorted by similarity to user profile
- **Filtering**: Optional exclusion of previously purchased items

## Performance Considerations

### Computational Complexity
- **TF-IDF Fitting**: O(n × m) where n = documents, m = vocabulary size
- **Embedding Generation**: O(n × m) for sparse matrix operations
- **Similarity Search**: O(log n) with HNSW indexing (ChromaDB)
- **User Profile Creation**: O(k) where k = number of purchased products

### Memory Usage
- **TF-IDF Vectors**: Sparse representation saves memory for large vocabularies
- **ChromaDB Storage**: Optimized for high-dimensional vectors
- **Embeddings**: Dense arrays for similarity calculations

### Scalability Factors
- **Vocabulary Growth**: TF-IDF dimensions increase with unique terms
- **Product Catalog Size**: Linear impact on storage and search time
- **User Base Growth**: Separate storage scaling for user profiles

## Troubleshooting

### Common Issues and Solutions

#### 1. Empty Recommendations
**Symptoms**: `recommend_products()` returns empty list
**Causes**:
- User profile doesn't exist
- No products in database
- All products already purchased (with exclude_purchased=True)

**Solutions**:
```python
# Check if user exists
user_result = rec_system.users_collection.get(ids=["user_id"])
if not user_result['embeddings']:
    print("User profile not found")

# Check product count
products = rec_system.products_collection.get()
print(f"Total products: {len(products['ids'])}")

# Disable purchase exclusion
recommendations = rec_system.recommend_products("user_id", exclude_purchased=False)
```

#### 2. Low Similarity Scores
**Symptoms**: All recommendations have very low similarity scores
**Causes**:
- Insufficient text in product descriptions
- No common vocabulary between user purchases and recommendations
- Overly generic product descriptions

**Solutions**:
- Enhance product descriptions with more specific terms
- Add product categories and features to descriptions
- Use domain-specific vocabulary

#### 3. Memory Issues with Large Catalogs
**Symptoms**: OutOfMemoryError during TF-IDF fitting
**Causes**:
- Very large vocabulary from extensive product descriptions
- Dense matrix operations on large datasets

**Solutions**:
```python
# Reduce vocabulary size
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,  # Limit vocabulary
    min_df=2,           # Increase minimum frequency
    max_df=0.95        # Remove very common terms
)

# Use sparse matrices
embeddings_sparse = tfidf.fit_transform(documents)
# Don't convert to dense array immediately
```

#### 4. Poor Recommendation Quality
**Symptoms**: Irrelevant or unexpected recommendations
**Causes**:
- Insufficient user purchase history
- Generic product descriptions
- Category mismatches

**Solutions**:
- Collect more user interaction data
- Implement category-based filtering
- Add user feedback mechanisms
- Weight recent purchases more heavily

### Debugging Tools

```python
# Check TF-IDF vocabulary
print("Vocabulary size:", len(rec_system.tfidf.vocabulary_))
print("Sample terms:", list(rec_system.tfidf.vocabulary_.keys())[:10])

# Examine user embedding
user_result = rec_system.users_collection.get(ids=["user_id"], include=['embeddings'])
user_vector = np.array(user_result['embeddings'][0])
print("User vector stats:")
print(f"Mean: {user_vector.mean():.4f}")
print(f"Std: {user_vector.std():.4f}")
print(f"Non-zero elements: {np.count_nonzero(user_vector)}")

# Check product similarity manually
from sklearn.metrics.pairwise import cosine_similarity
product1 = rec_system.products_collection.get(ids=["prod1"], include=['embeddings'])
product2 = rec_system.products_collection.get(ids=["prod2"], include=['embeddings'])
similarity = cosine_similarity([product1['embeddings'][0]], [product2['embeddings'][0]])[0][0]
print(f"Direct similarity: {similarity:.4f}")
```

## Next Steps and Improvements

### 1. Enhanced User Modeling
- **Implicit Feedback**: Track views, cart additions, time spent
- **Temporal Weighting**: Give more weight to recent interactions
- **Negative Feedback**: Learn from items users dislike or return
- **Multi-dimensional Preferences**: Separate models for different contexts

### 2. Advanced Text Processing
- **Semantic Embeddings**: Use BERT, Word2Vec, or FastText instead of TF-IDF
- **Multilingual Support**: Handle products in multiple languages
- **Entity Recognition**: Extract brands, specifications, and features
- **Sentiment Analysis**: Consider review sentiment in recommendations

### 3. Hybrid Recommendation Approaches
- **Collaborative Filtering**: Add user-user and item-item similarity
- **Matrix Factorization**: Implement SVD or NMF for latent factors
- **Deep Learning**: Neural collaborative filtering networks
- **Ensemble Methods**: Combine multiple recommendation algorithms

### 4. Real-time Capabilities
- **Streaming Updates**: Handle real-time user interactions
- **Incremental Learning**: Update models without full retraining
- **A/B Testing**: Compare recommendation algorithms
- **Online Learning**: Adapt to user feedback immediately

### 5. Business Logic Integration
- **Inventory Awareness**: Don't recommend out-of-stock items
- **Price Sensitivity**: Consider user's price preferences
- **Promotional Items**: Boost recommended products on sale
- **Business Rules**: Implement category restrictions or promotion rules

### 6. Evaluation and Metrics
- **Offline Metrics**: Precision, Recall, NDCG, diversity
- **Online Metrics**: Click-through rate, conversion rate, user satisfaction
- **Cross-validation**: Time-based splits for temporal data
- **Cold Start Analysis**: Performance for new users and products

### 7. Production Considerations
- **API Development**: RESTful endpoints for recommendations
- **Caching**: Redis for frequently accessed recommendations
- **Load Balancing**: Handle multiple concurrent users
- **Monitoring**: Track system performance and recommendation quality
- **Data Pipeline**: Automated data ingestion and model updates

### Sample Advanced Implementation

```python
class AdvancedRecommendationSystem(RecommendationSystem):
    def __init__(self):
        super().__init__()
        self.user_interactions = {}  # Track user behavior
        self.item_popularity = {}    # Track item popularity
        
    def add_interaction(self, user_id, product_id, interaction_type, timestamp):
        """Track user interactions for improved recommendations"""
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        
        self.user_interactions[user_id].append({
            'product_id': product_id,
            'type': interaction_type,  # view, cart, purchase, like
            'timestamp': timestamp,
            'weight': self._get_interaction_weight(interaction_type)
        })
    
    def _get_interaction_weight(self, interaction_type):
        """Assign weights to different interaction types"""
        weights = {
            'view': 1.0,
            'cart': 2.0,
            'purchase': 5.0,
            'like': 3.0,
            'dislike': -1.0
        }
        return weights.get(interaction_type, 1.0)
    
    def create_weighted_user_profile(self, user_id):
        """Create user profile with weighted interactions"""
        if user_id not in self.user_interactions:
            return
        
        interactions = self.user_interactions[user_id]
        weighted_embeddings = []
        total_weight = 0
        
        for interaction in interactions:
            product_embedding = self._get_product_embedding(interaction['product_id'])
            if product_embedding is not None:
                weight = interaction['weight']
                # Apply temporal decay
                days_ago = (datetime.now() - interaction['timestamp']).days
                temporal_weight = math.exp(-days_ago / 30)  # 30-day half-life
                
                final_weight = weight * temporal_weight
                weighted_embeddings.append(product_embedding * final_weight)
                total_weight += final_weight
        
        if weighted_embeddings and total_weight > 0:
            user_embedding = np.sum(weighted_embeddings, axis=0) / total_weight
            return user_embedding.tolist()
        return None
```

### Performance Monitoring

```python
class RecommendationMetrics:
    def __init__(self):
        self.metrics = {
            'total_recommendations': 0,
            'avg_response_time': 0,
            'cache_hit_rate': 0,
            'user_satisfaction': 0
        }
    
    def log_recommendation(self, user_id, recommendations, response_time):
        """Log recommendation event for analysis"""
        self.metrics['total_recommendations'] += 1
        self.metrics['avg_response_time'] = (
            (self.metrics['avg_response_time'] * (self.metrics['total_recommendations'] - 1) + response_time) /
            self.metrics['total_recommendations']
        )
    
    def evaluate_recommendations(self, user_id, recommendations, actual_interactions):
        """Evaluate recommendation quality"""
        if not recommendations or not actual_interactions:
            return 0
        
        recommended_ids = [r['product_id'] for r in recommendations]
        hit_count = len(set(recommended_ids) & set(actual_interactions))
        precision = hit_count / len(recommended_ids)
        recall = hit_count / len(actual_interactions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        }
```

This recommendation system provides a solid foundation for content-based product recommendations using modern vector database technology and proven machine learning techniques. The modular design allows for easy extension and improvement as your requirements evolve.
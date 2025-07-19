import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import requests
from io import BytesIO

class ImageSimilaritySearch:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)

        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )

    def encode_image(self, image_path_or_url):
        """Encode image to vector using CLIP"""
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path_or_url)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().tolist()[0]

    def encode_text(self, text):
        """Encode text to vector using CLIP"""
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().tolist()[0]

    def add_image(self, image_path_or_url, metadata=None):
        """Add image to the vector store"""
        try:
            embedding = self.encode_image(image_path_or_url)
            image_id = f"img_{len(self.collection.get()['ids'])}"

            self.collection.add(
                embeddings=[embedding],
                documents=[image_path_or_url],
                ids=[image_id],
                metadatas=[metadata or {}]
            )
            print(f"Added image: {image_path_or_url}")
        except Exception as e:
            print(f"Error adding image {image_path_or_url}: {e}")

    def search_by_image(self, image_path_or_url, n_results=3):
        """Find similar images"""
        query_embedding = self.encode_image(image_path_or_url)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def search_by_text(self, text, n_results=3):
        """Find images matching text description"""
        query_embedding = self.encode_text(text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

# Example usage
image_search = ImageSimilaritySearch()

# Sample image URLs (you can replace with local paths
sample_images = [
    "./cat.jpg",
    "./dog.jpg",
    "./car.jpg",
    "./tree.jpg"
]

# Add images to the collection
for img_url in sample_images:
    image_search.add_image(img_url, {"source": "sample_dataset"})

# Search by text
text_query = "a cute cat"
results = image_search.search_by_text(text_query)
print(f"Images matching '{text_query}':")
for doc in results['documents'][0]:
    print(f"- {doc}")
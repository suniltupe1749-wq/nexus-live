import os
import time  # Added for rate limiting
import chromadb
import google.generativeai as genai
from pymongo import MongoClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import uuid

# Load Secrets
# (Replace with your actual key if not using .env)
api_key = "AIzaSyA7gG30_Rb1d9zzOYK1jff6twf_eVKSbBY"
genai.configure(api_key=api_key)

class DatabaseManager:
    def __init__(self):
        # 1. MongoDB (Tables)
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            self.mongo_client.server_info()
            self.mongo_db = self.mongo_client["full_project_db"]
            self.table_col = self.mongo_db["tables"]
            print("✅ MongoDB Connected")
        except:
            print("⚠️ MongoDB not found. Tables will be text-only.")
            self.table_col = None

        # 2. ChromaDB (Vectors)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="enterprise_data",
            embedding_function=self.ef
        )

    def save_chunk(self, text, metadata):
        doc_id = str(uuid.uuid4())
        self.collection.add(documents=[text], metadatas=[metadata], ids=[doc_id])

    def save_table(self, html, summary, filename, page):
        mongo_id = "N/A"
        if self.table_col is not None:
            res = self.table_col.insert_one({"html": html, "source": filename, "page": page})
            mongo_id = str(res.inserted_id)
        
        meta = {"source": filename, "page": page, "type": "table", "mongo_id": mongo_id}
        self.save_chunk(f"Table Data: {summary}", meta)

    def ask_ai(self, user_query):
        # Pause briefly to avoid 429 errors
        time.sleep(1)

        # Search Database
        results = self.collection.query(query_texts=[user_query], n_results=5)
        
        context_parts = []
        retrieved_images = []
        retrieved_tables = []
        
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                dtype = meta.get('type')
                
                if dtype == 'image':
                    img_path = meta.get('image_path')
                    
                    # --- FIX 1: Prevent Duplicate Images ---
                    if img_path not in retrieved_images:
                        retrieved_images.append(img_path)
                        # We still add the text context so the AI knows about it
                        context_parts.append(f"[Found Image: {doc}]")

                elif dtype == 'table':
                    mongo_id = meta.get('mongo_id')
                    if self.table_col is not None and mongo_id != "N/A":
                        from bson.objectid import ObjectId
                        t_doc = self.table_col.find_one({"_id": ObjectId(mongo_id)})
                        
                        # --- FIX 2: Prevent Duplicate Tables ---
                        if t_doc and t_doc["html"] not in retrieved_tables:
                            retrieved_tables.append(t_doc["html"])
                            context_parts.append(f"[Found Table: {doc}]")
                else:
                    context_parts.append(f"{doc}")

        if not context_parts: return "No info found.", [], []

        # Generate Answer
        try:
            # Using 2.5 Flash as it is currently the most stable free model
            model = genai.GenerativeModel('gemini-flash-latest')
            prompt = f"Answer using this context:\n{chr(10).join(context_parts)}\n\nQuestion: {user_query}"
            response = model.generate_content(prompt)
            return response.text, retrieved_images, retrieved_tables
        except Exception as e:
            return f"Error: {e}", [], []

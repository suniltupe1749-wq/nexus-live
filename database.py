import os
import time
import chromadb
import google.generativeai as genai
from pymongo import MongoClient
from chromadb import Documents, EmbeddingFunction, Embeddings
import uuid

# 1. Setup Google Gemini
api_key = os.getenv("AIzaSyBUM7or5qz9DHa6I_ZezaAU0i26dIT9EDs")
genai.configure(api_key=api_key)

# 2. Define a Lightweight Embedding Function
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # Using the standard embedding model
        model = "models/embedding-001"
        embeddings = []
        for text in input:
            # Retry logic with SLOWER intervals to avoid 429 Errors
            for attempt in range(3):
                try:
                    response = genai.embed_content(
                        model=model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(response['embedding'])
                    time.sleep(1.5) # Wait 1.5 seconds between requests (CRITICAL FIX)
                    break
                except Exception as e:
                    time.sleep(3) # If error, wait 3 seconds before retry
        return embeddings

class DatabaseManager:
    def __init__(self):
        # MongoDB Connection
        try:
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri: 
                mongo_uri = "mongodb://localhost:27017/"
                
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            self.mongo_client.server_info() 
            self.mongo_db = self.mongo_client["full_project_db"]
            self.table_col = self.mongo_db["tables"]
            print("✅ MongoDB Connected")
        except:
            print("⚠️ MongoDB not found. Tables will be text-only.")
            self.table_col = None

        # ChromaDB Connection
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_store")
        self.ef = GeminiEmbeddingFunction()
        
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
            try:
                res = self.table_col.insert_one({"html": html, "source": filename, "page": page})
                mongo_id = str(res.inserted_id)
            except:
                pass
        
        meta = {"source": filename, "page": page, "type": "table", "mongo_id": mongo_id}
        self.save_chunk(f"Table Data: {summary}", meta)

    def ask_ai(self, user_query):
        # Search Vector DB
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
                    if img_path not in retrieved_images:
                        retrieved_images.append(img_path)
                        context_parts.append(f"[Found Image: {doc}]")

                elif dtype == 'table':
                    mongo_id = meta.get('mongo_id')
                    if self.table_col is not None and mongo_id != "N/A":
                        from bson.objectid import ObjectId
                        try:
                            t_doc = self.table_col.find_one({"_id": ObjectId(mongo_id)})
                            if t_doc and t_doc["html"] not in retrieved_tables:
                                retrieved_tables.append(t_doc["html"])
                                context_parts.append(f"[Found Table: {doc}]")
                        except:
                            pass
                else:
                    context_parts.append(f"{doc}")

        if not context_parts: return "No info found.", [], []

        # SWITCHED TO STABLE MODEL (gemini-1.5-flash)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Answer using this context:\n{chr(10).join(context_parts)}\n\nQuestion: {user_query}"
        
        try:
            response = model.generate_content(prompt)
            return response.text, retrieved_images, retrieved_tables
        except Exception as e:
            return f"AI Error: {e}", [], []

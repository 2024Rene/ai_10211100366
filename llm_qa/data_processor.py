import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class DataProcessor:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )
    
    def process_csv(self, file_path: str) -> FAISS:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to text chunks
        texts = []
        for _, row in df.iterrows():
            text = " ".join([f"{col}: {val}" for col, val in row.items()])
            texts.extend(self.text_splitter.split_text(text))
        
        # Create vector store
        vectorstore = FAISS.from_texts(texts, self.embeddings)
        return vectorstore
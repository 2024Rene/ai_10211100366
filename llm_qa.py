import streamlit as st
import pandas as pd
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def llm_qa_app():
    st.header("🤖 LLM Question Answering")
    
    # Sidebar configuration
    st.sidebar.subheader("Model Settings")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2"],
        help="Model used for text embeddings"
    )
    
    # Main content
    st.write("### Ghana Election Results Analysis")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload Ghana Election Results CSV", type=['csv'])
    
    if uploaded_file:
        try:
            # Load and display data preview
            df = pd.read_csv(uploaded_file)
            st.write("#### Data Preview")
            st.dataframe(df.head())
            
            # Initialize text processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            
            # Convert DataFrame to text
            texts = []
            for _, row in df.iterrows():
                text = " ".join([f"{col}: {val}" for col, val in row.items()])
                texts.extend(text_splitter.split_text(text))
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            
            # Create vector store
            with st.spinner("Processing data..."):
                vectorstore = FAISS.from_texts(texts, embeddings)
                st.success("Data processed successfully!")
            
            # Q&A Interface
            st.write("### Ask Questions")
            question = st.text_input(
                "Enter your question about the election results:",
                placeholder="e.g., What were the total votes in 2020?"
            )
            
            if question:
                with st.spinner("Searching..."):
                    # Perform similarity search
                    docs = vectorstore.similarity_search_with_score(question, k=3)
                    
                    # Display results
                    st.write("#### Relevant Information:")
                    for i, (doc, score) in enumerate(docs, 1):
                        with st.expander(f"Source {i} (Confidence: {score:.2f})"):
                            st.write(doc.page_content)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload the Ghana Election Results CSV file to begin.")

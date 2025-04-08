import streamlit as st
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

def init_embeddings():
    """Initialize the embedding model"""
    try:
        # Ensure we're using CPU and single thread
        device = 'cpu'
        torch.set_num_threads(1)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 1}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def process_data(df, text_splitter):
    """Process DataFrame and create text chunks"""
    texts = []
    try:
        for idx, row in df.iterrows():
            text = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            chunks = text_splitter.split_text(text)
            texts.extend(chunks)
        return texts
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return []

def llm_qa_app():
    st.header("🤖 LLM Question Answering")
    
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    
    # Main content
    st.write("### Ghana Election Results Analysis")
    
    # Sidebar settings
    with st.sidebar:
        st.subheader("Model Settings")
        chunk_size = st.slider("Chunk Size", 100, 1000, 500)
        chunk_overlap = st.slider("Chunk Overlap", 0, 100, 50)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Ghana Election Results CSV",
        type=['csv'],
        help="Upload the Ghana Election Results dataset"
    )
    
    if uploaded_file:
        try:
            # Load and display data preview
            df = pd.read_csv(uploaded_file)
            with st.expander("Data Preview", expanded=False):
                st.dataframe(df.head())
            
            # Process data if not already done
            if st.session_state.vectorstore is None:
                with st.spinner("Processing data..."):
                    # Initialize components
                    if st.session_state.embeddings is None:
                        st.session_state.embeddings = init_embeddings()
                    
                    if st.session_state.embeddings:
                        # Text splitting
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            separators=["\n\n", "\n", ". ", ", "]
                        )
                        
                        # Process data in background
                        texts = process_data(df, text_splitter)
                        
                        if texts:
                            # Create vector store
                            try:
                                st.session_state.vectorstore = FAISS.from_texts(
                                    texts,
                                    st.session_state.embeddings,
                                    metadatas=[{"source": f"row_{i}"} for i in range(len(texts))]
                                )
                                st.success("✅ Data processed successfully!")
                            except Exception as e:
                                st.error(f"Error creating vector store: {str(e)}")
            
            # Enhanced Q&A Interface
            st.write("### Ask Questions")
            
            # Create three columns for better layout
            col1, col2, col3 = st.columns([3, 1, 1])#)
            
            with col1:
                question = st.text_input(
                    "Enter your question:",
                    placeholder="e.g., What were the total votes in 2020?",
                    key="question_input"
                )
            
            with col2:
                k = st.number_input("Number of results", 1, 5, 3)
                
            with col3:
                confidence_threshold = st.slider(
                    "Min. Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
            
            if question and st.session_state.vectorstore:
                with st.spinner("🔍 Searching for relevant information..."):
                    try:
                        # Perform similarity search
                        docs = st.session_state.vectorstore.similarity_search_with_score(
                            question, k=k
                        )
                        
                        # Create metrics container
                        metrics_container = st.container()
                        
                        with metrics_container:
                            # Display search metrics
                            st.write("#### Search Metrics")
                            metric_cols = st.columns(3)
                            with metric_cols[0]:
                                st.metric("Total Results", len(docs))
                            with metric_cols[1]:
                                avg_confidence = 1 - sum(score for _, score in docs) / len(docs)
                                st.metric("Avg. Confidence", f"{avg_confidence:.2%}")
                            with metric_cols[2]:
                                max_confidence = 1 - min(score for _, score in docs)
                                st.metric("Best Match", f"{max_confidence:.2%}")
                        
                        # Display results
                        st.write("#### Relevant Information")
                        
                        for i, (doc, score) in enumerate(docs, 1):
                            confidence = 1 - score  # Convert distance to confidence
                            
                            # Skip results below confidence threshold
                            if confidence < confidence_threshold:
                                continue
                                
                            with st.expander(
                                f"🔍 Result {i} (Confidence: {confidence:.2%})",
                                expanded=i==1
                            ):
                                # Display content in a formatted box
                                st.markdown("**Content:**")
                                st.code(doc.page_content, language="text")
                                
                                # Display metadata
                                st.markdown("**Metadata:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.info(f"Source: {doc.metadata['source']}")
                                with col2:
                                    st.info(f"Confidence Score: {confidence:.2%}")
                        
                        # No results above threshold
                        if all(1 - score < confidence_threshold for _, score in docs):
                            st.warning("No results met the confidence threshold. Try lowering the threshold or rephrasing your question.")
                        
                    except Exception as e:
                        st.error(f"Error performing search: {str(e)}")
                        st.info("💡 Tip: Try rephrasing your question or adjusting the confidence threshold.")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Clear data button - Updated implementation
    if st.session_state.vectorstore is not None:
        if st.sidebar.button("Clear Data"):
            # Clear session state
            for key in ['vectorstore', 'embeddings']:
                if key in st.session_state:
                    del st.session_state[key]
            # Use the new rerun method
            st.rerun()
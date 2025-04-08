import streamlit as st
import os

# Configure environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure Streamlit
st.set_page_config(
    page_title="AI Exam App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try importing torch with better error handling
try:
    import torch
    torch.set_num_threads(1)
    st.sidebar.success("PyTorch version: " + torch.__version__)
except ImportError as e:
    st.error(f"""
    Failed to import PyTorch. Please install it using:
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
    Error: {str(e)}
    """)
    st.stop()

# Import modules after configuration
from my_modules.regression import regression_app
from my_modules.clustering import clustering_app
from my_modules.neural_net import neural_net_app
from my_modules.llm_qa import llm_qa_app

st.title("🧠 AI Exam Dashboard")

app_mode = st.sidebar.selectbox("Choose Module", [
    "Regression", "Clustering", "Neural Network", "LLM Q&A"
])

if app_mode == "Regression":
    regression_app()
elif app_mode == "Clustering":
    clustering_app()
elif app_mode == "Neural Network":
    neural_net_app()
elif app_mode == "LLM Q&A":
    llm_qa_app()

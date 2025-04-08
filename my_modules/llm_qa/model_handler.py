from typing import Dict, List
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class ModelHandler:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain LLM
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
    
    def create_qa_chain(self, vectorstore) -> RetrievalQA:
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        return qa_chain
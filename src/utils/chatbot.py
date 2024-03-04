import gradio as gr 
import time 
import openai 
import os 
from langchain.vectorstores import chroma
from typing import List, Tuple 
import re 
import ast
import html
from utils.load_config import LoadConfig
APPCFG = LoadConfig()
URL = "https://github.com/amirshq/GPT-RAG"

class ChatBot:
    @staticmethod
    def respond(chatbot: List, message: str,data_type:str = "Preprocessed doc", temperature: float = 0.0) -> Tuple:
        if data_type == "Preprocessed doc":
            # directories
            if os.path.exists(APPCFG.persist_directory):
                vectordb = Chroma(persist_directory=APPCFG.persist_directory,
                                  embedding_function=APPCFG.embedding_model)
            else:
                chatbot.append(
                    (message, f"VectorDB does not exist. Please first execute the 'upload_data_manually.py' module."))
                return "", chatbot, None
        elif data_type == "Upload doc: Process for RAG":
            if os.path.exists(APPCFG.custom_perist_directory):
                vectordb = Chroma(persist_directory=APPCFG.custom_perist_directory,
                                  embedding_function=APPCFG.embedding_model)
            else:
                chatbot.append(
                    (message, f"VectorDB does not exist. Please first execute the 'upload' button."))
                return "", chatbot, None
    
            )
        docs = vectordb.similarity_search(message, k=APPCFG.k)
        question = 
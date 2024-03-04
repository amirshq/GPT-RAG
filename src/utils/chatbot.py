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

"""
    This module contains the ChatBot class which is responsible for responding to user queries.
    This code defines a part of a chatbot system, focusing on a method that responds to user 
    messages based on the type of document interaction requested. The method decides what action 
    to take based on whether the user wants to work with a preprocessed document or upload a document 
    for processing. 
    It checks if certain directories exist on the file system
    (to confirm if the necessary data or configuration is available) and uses these checks 
    to initialize a vector database for processing the documents, or informs the user if 
    the required setup is not found, suggesting steps to rectify the situation. 
    Essentially, this method is about setting up the right environment for 
    document processing based on user requests and the chatbot's configuration.
"""
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
        question = "#User new question\n" + message 
        
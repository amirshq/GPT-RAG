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
        
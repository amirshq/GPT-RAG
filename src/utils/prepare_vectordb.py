from langchain.vectorstores import Chroma
from langchain.document_loaders import pyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import list 
from langchain.embeddings.openai import OpenAIEmbeddings

class PrepareVectorDB:
    def __init__(
            self,
            data_directory:str,
            persist_directory:str,
            embedding_model_engine:str,
            chunk_size:int, 
            chunk_overlap:int
    ) -> None:
        
        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n","\n"," ",""]
            
        )
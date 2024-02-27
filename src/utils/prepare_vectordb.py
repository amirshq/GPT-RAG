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
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings()
    def __load_all_documents(self) -> list: 
        doc_counter = 0
        if isinstance(self.data_directory,list):
            print("Loading the uploaded documents...")
            docs = []
            for doc_dir in self.data_directory:
                docs.extend(pyPDFLoader(doc_dir).load())
                doc_counter +=1
            print("Number of loaded documents: ",doc_counter)
            print("Number of pages:",len(docs),"\n\n")
        else:
            print("Loading douments Manually...")
            document_list = os.listdir(self.data_directory)
            docs = []
            for doc_name in document_list:
                docs.extend(pyPDFLoader(os.path.join(
                    self.data_directory,doc_name)).load())
                doc_counter += 1 
            print("Number of Loaded Documents:", doc_counter)
            print("Number of pages:" len(docs),"\n\n")
        return docs
    def 
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
    def __chunk_documents(self,docs:List) -> List:
        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:",len(chunked_documents),"\n\n")
        return chunked_documents
    def prepare_and_save_vectordb(self):
        docs = self.__load_all_documents()
        chunked_documents = self.__chunk_documents(docs)
        print("Preparing vectordb...")
        vectordb = Chroma.from_documents(
            documents = chunked_documents,
            embedding = self.embedding,
            Persist_directory = self.persist_directory
        )
        print("VectorDB is created and saved.")
        print("Number os vectors in vectordb:"
            vectordb.__collection.count(),"\n\n")
        return vectordb
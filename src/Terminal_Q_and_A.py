"""
    this code snippet demonstrates an interactive Q&A system where
    users can input questions, and the system retrieves relevant 
    information from a vector database and generates responses using 
    an AI language model provided by OpenAI.
"""
import openai 
import yaml 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import chroma
from typing  import list, Tuple 
from utils.load_config import LoadConfig

#Loading openai credentials
APPCFG = LoadConfig()

with open("configs/app_config.yml") as cfg:
    app_config= yaml.load(cfg,loader=yaml.FullLoader)

#Load the mebedding function
embedding = OpenAIEmbeddings()
#Load the vector database
vectordb = chroma(persist_directory=APPCFG.persist_directory,embedding_function=embedding)

print("Number of vectors in the vector database: ",vectordb._collection.count())

#Prepare the RAG with OpenAI in terminal 
while True:
    question = input("Ask a question or press 'q' to exit: ")
    if question.lower() == 'q':
        break
    question = "#User new question\n" + question
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrieved_docs_page_content: List[Tuple] = [
        
    ]
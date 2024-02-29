import openai 
import os 
from dotenv import load_dotenv
import yaml 
from langchain.embeddings.openai import OpenAIEmbeddings
from pyprojroot import here
import shutil

load_dotenv()

class LoadConfig:
    def __init__(self) -> None:
        with open(here("configs/app_config.yml")) as cfg:
            app_config= yaml.load(cfg,loader=yaml.FullLoader)
        #LLM configs
        self.llm_engine = app_config["llm_config"]["engine"]
        self.llm_system_role = app_config["llm_config"]["llm_system_role"]
        self.persist_directory = str(here(
            app_config["directories"]["persist_directory"]))
        self.custom_perist_directory = str(here(
            app_config["directories"]["custom_persist_directory"]))
        self.embedding_model = OpenAIEmbeddings()

        #Retrieval Configs 
        self.data_directory = app_config["directories"]["data_directory"]
        self.k = app_config["retrieval_config"]["k"]
        self.embedding_model_engine = app_config["embedding_model_config"]["engine"]
        self.chunk_size = app_config["splitter_config"]["chunk_size"]
        self.chunk_overlap = app_config["splitter_config"]["chunk_overlap"]

        #summarization configs
        self.max_final_token = app_config["summarizer_config"]["max_final_token"]
        self.character_overlap = app_config["summarizer_config"]["character_overlap"]
        self.token_threshold = app_config["summarizer_config"]["token_threshold"]
        self.summarizer_llm_system_role = app_config["summarizer_config"]["summarizer_llm_system_role"]
        self.final_summarizer_llm_system_role = app_config["summarizer_config"]["final_summarizer_llm_system_role"]
        self.temperature = app_config["llm_config"]["temperature"]

        #memory configs
        self.number_of_q_a_pairs = app_config["memory"]["number_of_q_a_pairs"]

        #Load OpenAI API Key
        self.load_openai_cfg()

        
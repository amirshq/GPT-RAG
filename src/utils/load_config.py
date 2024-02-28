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
        
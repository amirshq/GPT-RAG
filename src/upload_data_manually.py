import os
from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig
CONFIG = LoadConfig()

def upload_data_manually() -> None:
    """
    Prepares an instance of PrepareVectorDB.
    Manually transfers data to the VectorDB.

    This method begins by creating an instance of PrepareVectorDB, 
    setting it up with key parameters like data_directory, persist_directory, 
    embedding_model_engine, chunk_size, and chunk_overlap. 
    It subsequently verifies whether the VectorDB already resides in 
    the specified persist_directory. If absent, it proceeds to invoke 
    the prepare_and_save_vectordb function to generate and store the VectorDB. 
    In case the VectorDB already exists, a notification is printed to 
    confirm its presence.
    
    Args:
        data_directory (str): Directory containing the data.
        persist_directory (str): Directory to persist prepared VectorDB.
        embedding_model_engine (str): Engine for embedding model.
        chunk_size (int): Size of data chunks.
        chunk_overlap (int): Overlap size between data chunks.

    Returns:
        None
    """
    prepare_vectordb_instance = PrepareVectorDB(
        data_directory=CONFIG.data_directory,
        persist_directory=CONFIG.persist_directory,
        embedding_model_engine=CONFIG.embedding_model_engine,
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )
    if not len(os.listdir(CONFIG.persist_directory)) !=0:
        prepare_vectordb_instance.prepare_and_save_vectordb()
    else:
        print(f"VectorDB already exist in {CONFIG.persist_directory}")
    return None

if __name__ == "__main__":
    upload_data_manually()
################################################################################################
# SETTINGS
################################################################################################
import yaml
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

def get_settings(model_type: str="small"):
    if model_type not in ["small", "big"]:
        raise ValueError("model_type must be either 'small' or 'big'")
    
    # 0. 讀取 config.yml
    with open("/home/End_to_End_RAG/config.yml", "r") as f:
        config = yaml.safe_load(f)

    Settings.llm = Ollama(
        model=config["model"]["llm_model"], 
        base_url=config["model"]["ollama_url"],
        request_timeout=300.0
    )
    
    Settings.eval_llm = Ollama(
        model=config["eval_model"]["llm_model"], 
        base_url=config["eval_model"]["ollama_url"],
        request_timeout=300.0
    )
    
    Settings.builder_llm = Ollama(
        model=config["builder_model"]["llm_model"], 
        base_url=config["builder_model"]["ollama_url"],
        request_timeout=300.0
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=config["model"]["embed_model"],
        base_url=config["model"]["ollama_url"],
        embed_batch_size=1,
        embed_batch_timeout=300.0,
    )

    Settings.lightrag_storage_path_DIR = config["lightrag"]["lightrag_storage_path_DIR"]
    Settings.lightrag_language = config["lightrag"]["lightrag_language"]
    Settings.lightrag_entity_types = config["lightrag"]["lightrag_entity_types"]
    Settings.raw_file_path_DI = config["data"]["raw_file_path_DI"]
    Settings.raw_file_path_GEN = config["data"]["raw_file_path_GEN"]
    Settings.qa_file_path_DI = config["data"]["qa_file_path_DI"]
    Settings.qa_file_path_GEN = config["data"]["qa_file_path_GEN"]

    return Settings
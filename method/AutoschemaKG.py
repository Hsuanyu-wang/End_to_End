from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI

# 1. 將模型名稱改為您在 Ollama 中 pull 的模型名稱 (例如：llama3.1 或 llama3.1:8b)
model_name = "qwen2.5:7b" 
# 2. 將 client 改為連線至本機的 Ollama API
client = OpenAI(
    base_url='http://192.168.63.174:11434/v1',
    api_key='ollama' # Ollama 不需要真實的 API Key，但 openai 套件會檢查此欄位，填入任意字串即可
)
keyword = 'schema'
output_directory = f'/home/End_to_End_RAG/method/{keyword}'
# 3. 將 client 與 model_name 傳入 LLMGenerator
triple_generator = LLMGenerator(client, model_name=model_name)
kg_extraction_config = ProcessingConfig(
      model_path=model_name,
      data_directory="/home/End_to_End_RAG/AutoschemaKG/schema_learning",
      filename_pattern="schema_learning.jsonl", # Will read the files with string filename_patterns in the data directory as input files
      batch_size_triple=1, # batch size for triple extraction
      batch_size_concept=32, # batch size for concept generation
      output_directory=f"{output_directory}",
      max_new_tokens=2048,
      max_workers=3,
      remove_doc_spaces=True, # For removing duplicated spaces in the document text
)

kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)

# Construct entity&event graph
kg_extractor.run_extraction() # Involved LLM Generation
# Convert Triples Json to CSV
kg_extractor.convert_json_to_csv()
# Concept Generation
kg_extractor.generate_concept_csv_temp(batch_size=64) # Involved LLM Generation
# Create Concept CSV
kg_extractor.create_concept_csv()
# Convert csv to graphml for networkx
kg_extractor.convert_to_graphml()
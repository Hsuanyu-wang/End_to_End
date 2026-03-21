from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI
from transformers import pipeline
# client = OpenAI(api_key='<your_api_key>',base_url="<your_api_base_url>") 
# model_name = "meta-llama/llama-3.1-8b-instruct"

model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",
)
keyword = 'Dulce'
output_directory = f'import/{keyword}'
triple_generator = LLMGenerator(client, model_name=model_name)
kg_extraction_config = ProcessingConfig(
      model_path=model_name,
      data_directory="example_data",
      filename_pattern=keyword, # Will read the files with string filename_patterns in the data directory as input files
      batch_size_triple=3, # batch size for triple extraction
      batch_size_concept=16, # batch size for concept generation
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
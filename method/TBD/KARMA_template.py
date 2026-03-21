from karma import KARMAPipeline
from karma.config import create_default_config

# Create configuration
config = create_default_config(api_key="your-openai-api-key")

# Initialize pipeline
pipeline = KARMAPipeline.from_config(config)

# Process document
result = pipeline.process_document("path/to/document.pdf")

# Access results
print(f"Extracted {len(result.integrated_triples)} knowledge triples")
for triple in result.integrated_triples[:5]:
    print(f"{triple.head} --[{triple.relation}]--> {triple.tail} (confidence: {triple.confidence:.2f})")

# Export knowledge graph
pipeline.export_knowledge_graph("knowledge_graph.json")
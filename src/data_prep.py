import os
from dotenv import load_dotenv

# LlamaIndex Core Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Settings # <-- CRITICAL IMPORT

# Ollama Specific Imports (ensure these are installed: pip install llama-index-llms-ollama llama-index-embeddings-ollama)
from llama_index.llms.ollama import Ollama as OllamaLLM
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Setup ---

load_dotenv()
os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-run"
# We set a dummy key because LlamaIndex checks for this env var even when using Ollama

DATA_DIR = "product_data"
TOOL_NAME = "product_data_retriever"

def setup_llamaindex_tool() -> QueryEngineTool:
    """
    Loads documents, creates a VectorStoreIndex, and exposes it as a QueryEngineTool
    for LangChain agents.
    """

    # CRITICAL: Configure LlamaIndex to use Ollama models globally
    print("--- Configuring LlamaIndex Settings for Ollama ---")
    # Set the LLM for query synthesis/summarization
    Settings.llm = OllamaLLM(model="llama3", request_timeout=120) 
    
    # Set the embedding model for chunking/indexing (llama3 can serve embeddings too)
    # Note: OllamaEmbedding uses the same model name by default for embeddings
    Settings.embed_model = OllamaEmbedding(model_name="llama3")

    print("Settings updated. Proceeding with data ingestion.")    
    print(f"--- LlamaIndex configured to use Ollama/llama3 for LLM and Embeddings ---")


    print(f"--- 1. Loading documents from '{DATA_DIR}/' ---")
    # 1. Load Data: Use the SimpleDirectoryReader to find all files in the directory.
    documents = SimpleDirectoryReader(input_dir=DATA_DIR).load_data()
    print(f"Successfully loaded {len(documents)} documents.")

    print("--- 2. Creating Vector Store Index (Chunking & Embedding) ---")
    # 2. Create Index: This is the critical step. 
    # It chunks the documents, generates embeddings for each chunk (using OpenAI), 
    # and stores them in a Vector Store (in-memory for this tutorial).
    index = VectorStoreIndex.from_documents(documents)
    print("Indexing complete.")

    # 3. Create Query Engine and Tool:
    # A QueryEngine is a high-level abstraction for querying the index.
    query_engine = index.as_query_engine()

    # QueryEngineTool is a LlamaIndex class that wraps the query engine 
    # to be easily consumable by a LangChain Agent.
    product_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            # The name is critical! This is what the LangChain Agent will call.
            name=TOOL_NAME,
            # The description guides the LLM on WHEN to use the tool.
            description=(
                "Use this tool to retrieve factual data on the product 'SynapseFlow', "
                "including technical specs, brand voice rules, and past marketing examples. "
                "Always use this before writing product copy."
            ),
        ),
    )
    
    print(f"--- 3. LlamaIndex tool '{TOOL_NAME}' created successfully ---")
    return {
        "query_engine": query_engine,
        "name": "product_data_retriever", # This is the 'product_data_retriever' 
        "description": ("Use this tool to retrieve factual data on the product 'SynapseFlow', "
                        "including technical specs, brand voice rules, and past marketing examples. "
                        "Always use this before writing product copy."),
    }

if __name__ == "__main__":
    # Test the index creation and a simple retrieval query
    tool = setup_llamaindex_tool()
    
    # Test the underlying query engine
    print("\n--- Testing the Query Engine directly ---")
    test_query = "What is the key metric improvement of Adaptive Noise Cancellation 2.0 and why is it zero latency?"
    
    # We call the underlying query engine directly here for a quick test
    response = tool["query_engine"].query(test_query)
    
    print(f"\nUser Query: {test_query}")
    print("\nLLaMAIndex Response:")
    print(response) 
    
    # Expected output should mention 35% COI reduction and local processing.
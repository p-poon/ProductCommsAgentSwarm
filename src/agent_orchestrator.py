import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

# LlamaIndex Tool Import
from .data_prep import setup_llamaindex_tool

# --- Setup ---
load_dotenv()

LLM = Ollama(model="llama3", temperature=0.5)

# Load the LlamaIndex Tool (query_engine)
raw_tool_data = setup_llamaindex_tool()
query_engine = raw_tool_data["query_engine"]

def retrieve_product_context(feature_name: str) -> str:
    """
    Uses the LlamaIndex query engine to retrieve relevant product context for the feature.
    """
    retrieval_query = f"Provide all available product data and key metrics for the feature: {feature_name}."
    result = query_engine.query(retrieval_query)
    # If result is an object, convert to string as needed
    if hasattr(result, "response"):
        return str(result.response)
    return str(result)

def run_communications_workflow(feature_name: str) -> dict:
    """
    Generates communication materials for a product feature using retrieval-augmented LLM chains.
    """
    print(f"\n{'='*50}\nRUNNING WORKFLOW FOR FEATURE: {feature_name}\n{'='*50}")

    # --- 1. Retrieve context ---
    product_context = retrieve_product_context(feature_name)
    print("\n--- Retrieved Product Context ---")
    print(product_context)

    # --- 2. Social Media Copy ---
    social_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are the **SynapseFlow Social Media Copywriter**. Your goal is to generate "
         "a concise, exciting, launch-ready tweet (max 280 characters). "
         "Your tone must be **Enthusiastic** and **Forward-Thinking**. "
         "End with exactly one relevant emoji and the hashtag #SynapseFlow. "
         "Focus on the *benefit* to the user, not just the technical spec. "
         "Use the following product context to inform your writing:\n\n{context}"
        ),
        ("human", "{input}")
    ])
    social_chain = LLMChain(llm=LLM, prompt=social_prompt)
    social_input = f"Generate a launch announcement tweet for the feature: {feature_name}."
    social_post = social_chain.invoke({"input": social_input, "context": product_context})["text"]

    # --- 3. Documentation Copy ---
    doc_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are the **SynapseFlow Technical Documentation Specialist**. Your goal is to write "
         "a detailed, factual, and neutral FAQ answer. "
         "Your tone must be **Clear and Objective**. "
         "Present all specifications accurately using a **bulleted list** format. "
         "Do not use emojis or marketing hype. "
         "Use the following product context to inform your answer:\n\n{context}"
        ),
        ("human", "{input}")
    ])
    doc_chain = LLMChain(llm=LLM, prompt=doc_prompt)
    doc_input = f"Write a comprehensive FAQ answer explaining the details of the {feature_name} feature."
    faq_entry = doc_chain.invoke({"input": doc_input, "context": product_context})["text"]

    return {
        "feature_name": feature_name,
        "social_media_post": social_post,
        "faq_answer": faq_entry
    }

if __name__ == "__main__":
    # Define the product feature you want communications for
    target_feature = "Adaptive Noise Cancellation 2.0 (ANC 2.0)"
    
    # Run the orchestration
    results = run_communications_workflow(target_feature)

    # Print Final Results
    print(f"\n\n{'*'*60}")
    print(f"COMMUNICATIONS MATERIALS GENERATED FOR: {results['feature_name']}")
    print(f"{'*'*60}\n")
    
    print(">>> 📣 TWITTER POST (SocialMediaAgent):")
    print(results['social_media_post'])
    print("\n" + "-"*30 + "\n")
    
    print(">>> 📋 USER DOCUMENTATION (DocumentationAgent):")
    print(f"Q: How does the new {results['feature_name']} work?")
    print(f"A: {results['faq_answer']}")
#    print(f"\n\nNote: This version uses retrieval-augmented LLM chains and does not require agent_scratchpad or ReAct logic.")
# 💬 The Product Communications Agent Swarm

The **Product Communications Agent Swarm** is a Retrieval-Augmented Generation (RAG) system built using **LlamaIndex** and **LangChain** to efficiently generate varied communication materials—from short social media copy to detailed documentation—all based on a single source of truth (your internal product documents).

## 💡 Project Architecture & Goal

The core concept is to use a team of specialized **LangChain Agents**, each with a distinct persona and tone, to query a single, unified data index managed by **LlamaIndex**.

| Framework | Role | Output |
| :--- | :--- | :--- |
| **LlamaIndex** | **Data Expert (RAG Pipeline)**: Ingests unstructured product specs and brand guides, creates vector embeddings using a **local Ollama model**, and exposes a unified `product_data_retriever` tool. | Accurate, retrieved facts/metrics. |
| **LangChain** | **Orchestrator (Agent Swarm)**: Defines two specialized ReAct agents (`SocialMediaAgent` and `DocumentationAgent`), provides them with distinct personas, and executes them in sequence to generate targeted copy. | Finished, channel-ready communication materials. |

## ✨ Features

  * **100% Local LLM:** Uses **Ollama** (`llama3`) for both vector embedding and agent reasoning, eliminating cloud API costs and quotas.
  * **Persona-Driven Generation:** Agents adopt specific tones (e.g., *Enthusiastic Hype* vs. *Objective Technical*) based on custom system instructions.
  * **Tool Integration:** Seamlessly stitches the LlamaIndex retrieval engine into the LangChain Agent's decision-making loop.

-----

## 💻 Setup and Installation

### 1\. Prerequisites

You must have **Ollama** installed and running to use the local LLM and embedding model.

1.  **Install Ollama:** Follow the instructions for your OS on the [Ollama website](https://ollama.com).
2.  **Download Llama 3:** Open your terminal and download the model:
    ```bash
    ollama run llama3
    ```
3.  **Create Environment File:** Create a file named `.env` in the project root. Since we are using Ollama, we only need to provide a placeholder for `OPENAI_API_KEY` to prevent external libraries from defaulting:
    ```bash
    # .env
    OPENAI_API_KEY="DUMMY_KEY_FOR_LOCAL_LLM"
    ```

### 2\. Project Structure

Ensure your project matches this directory layout:

```
product-comm-agent-swarm/
├── product_data/                  # <- Your input files must be here
│   ├── brand_voice.txt
│   ├── marketing_copy_examples.txt
│   └── product_specs.txt
|
├── src/
│   ├── __init__.py                # Must be present for module imports
│   ├── data_prep.py               # LlamaIndex setup
│   └── agent_orchestrator.py      # LangChain agents & execution
|
├── .env                           # API key (ignored by Git)
└── requirements.txt               # Dependencies
```

### 3\. Install Dependencies

Activate your virtual environment and install all required Python packages:

```bash
# 1. Install required packages
pip install -r requirements.txt

# --- requirements.txt should contain (at minimum): ---
# python-dotenv
# langchain
# llama-index
# langchain-community
# langchain-ollama
# llama-index-llms-ollama
# llama-index-embeddings-ollama
```

-----

## 🚀 How to Run the Workflow

You must execute the main script as a **Python module** from the project's root directory to ensure the internal relative imports work correctly.

### Execution Command

```bash
python -m src.agent_orchestrator
```

### Expected Output

The console will print verbose logs (`verbose=True`) showing the following sequence for each agent:

1.  **START:** The `run_communications_workflow` begins.
2.  **LlamaIndex Setup:** The `data_prep.py` script loads data and configures **Ollama**.
3.  **Agent Thought Loop:**
      * **Thought:** The agent decides it needs to use the `product_data_retriever` tool.
      * **Action:** The agent queries the tool (which searches the LlamaIndex vector store).
      * **Observation:** The tool returns the relevant facts (e.g., "35% COI reduction").
      * **Final Answer:** The agent uses the facts and its persona prompt to generate the final copy.
4.  **Final Result:** The formatted **Twitter Post** and **Documentation FAQ** are printed.

-----

## Acknowledgements to AI Tools
* Google Gemini for the tutorial
* VSCode AI co-pilot for debugging and autocompletions

# Graph RAG System (DocuNet)

A lightweight and efficient Graph Retrieval-Augmented Generation (RAG) application. This system integrates LangGraph for stateful agentic workflows, Neo4j for structured knowledge graph storage, Streamlit for an interactive user interface, and Groq's fast LLM API for extraction and reasoning.

## Features

- **Document Ingestion:** Upload `.txt` or `.md` files. The system automatically chunks the text and extracts key entities and relationships using Groq's LLMs.
- **Knowledge Graph Integration:** Stores extracted entities and relationships directly into a Neo4j graph database.
- **Agentic Workflow:** Utilizes LangGraph to orchestrate a multi-step retrieval pipeline:
  1. Extract core entities from the user's question.
  2. Retrieve a 1-hop context neighborhood from Neo4j based on the extracted entities.
  3. Generate a comprehensive answer using the retrieved graph context.
- **Fast Inference:** Uses Groq's high-speed API with LLaMA 3 models (`llama3-8b` for extraction, `llama3-70b` for synthesis).
- **Interactive UI:** Built with Streamlit for easy setup, ingestion progress tracking, and chat interface.

## Project Structure

- `app.py`: The main Streamlit application script containing the frontend UI, file uploading logic, and chat interface.
- `ingestion_agent.py`: Handles text chunking and uses LangChain/Groq to extract Pydantic schema-defined entities and relationships from text.
- `rag_agent.py`: Defines the LangGraph workflow (`extract_entities` -> `retrieve_context` -> `generate_answer`).
- `graph_utils.py`: Contains the `Neo4jGraph` class to manage connections and Cypher queries with the Neo4j database.
- `requirements.txt`: Python dependencies required to run the project.
- `.env`: Configuration file for API keys and database URIs.

## Prerequisites

- **Python 3.8+**
- **Neo4j Database:** You need a running instance of Neo4j. You can use [Neo4j Desktop](https://neo4j.com/download/) locally or a free cloud instance on [Neo4j AuraDB](https://neo4j.com/cloud/aura/).
- **Groq API Key:** Sign up at [Groq](https://console.groq.com/) to get an API key.

## Setup Instructions

**1. Clone the repository and navigate to the project directory:**
```bash
cd DocuNet
```

**2. Activate the virtual environment:**
If you are using the existing virtual environment:
- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Linux/Mac:** `source venv/bin/activate`

*(If you need to create a new one, run `python -m venv venv` first)*

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables:**
Create a `.env` file in the root directory (or use the existing one) and add your credentials:
```env
NEO4J_URI=bolt://localhost:7687  # Update if using AuraDB
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
GROQ_API_KEY=your_groq_api_key
```

**5. Run the Application:**
Start the Streamlit development server:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Usage

1. **Setup & Connection:** Ensure the sidebar shows a successful connection to Neo4j.
2. **Document Ingestion:** Upload a text document via the sidebar. Wait for the extraction and writing processes to complete.
3. **Chat:** Ask questions about the ingested data in the main chat interface. You can expand the "View Retrieved Graph Context" to see the specific entities extracted from your prompt and the context retrieved from Neo4j.

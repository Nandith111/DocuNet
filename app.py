import streamlit as st
import os
from dotenv import load_dotenv
from graph_utils import Neo4jGraph
from ingestion_agent import chunk_text, extract_graph_from_text
from rag_agent import rag_agent_app

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Graph RAG System", layout="wide")
st.title("Graph RAG System with Neo4j & Groq")

# Sidebar for Setup and Ingestion
with st.sidebar:
    st.header("1. Setup & Connection")
    
    # Initialize connection
    graph = Neo4jGraph()
    if graph.driver:
        st.success("Connected to Neo4j!")
        stats = graph.get_all_nodes_summary()
        st.write(f"**Graph Stats:** {stats['nodes']} Nodes, {stats['relationships']} Relationships")
    else:
        st.error("Failed to connect to Neo4j. Check your .env file.")
        
    st.header("2. Document Ingestion")
    uploaded_file = st.file_uploader("Upload a text document", type=["txt", "md"])
    
    if st.button("Ingest Document") and uploaded_file:
        if not graph.driver:
            st.error("Cannot ingest: Neo4j is not connected.")
        else:
            with st.spinner("Processing document..."):
                text = uploaded_file.read().decode("utf-8")
                chunks = chunk_text(text, chunk_size=1500)
                
                total_entities = 0
                total_rels = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, chunk in enumerate(chunks):
                    status_text.text(f"Extracting from chunk {i+1}/{len(chunks)}...")
                    graph_data = extract_graph_from_text(chunk)
                    
                    # Pydantic models to dicts
                    entities_dicts = [e.model_dump() for e in graph_data.entities]
                    rels_dicts = [r.model_dump() for r in graph_data.relationships]
                    
                    if entities_dicts:
                        status_text.text(f"Writing chunk {i+1}/{len(chunks)} to Neo4j...")
                        graph.add_document_graph(entities_dicts, rels_dicts)
                        total_entities += len(entities_dicts)
                        total_rels += len(rels_dicts)
                    
                    progress_bar.progress((i + 1) / len(chunks))
                
                status_text.text("Done!")
                st.success(f"Ingestion complete! Added ~{total_entities} entities and ~{total_rels} relationships.")
            
    if st.button("Clear Chat"):
        st.session_state.messages = []

# Main Chat Interface
st.header("3. Chat with Graph RAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("View Retrieved Graph Context"):
                st.write(message["context"])

if prompt := st.chat_input("Ask a question about the knowledge graph..."):
    if not graph.driver:
        st.error("Neo4j is not connected. Cannot query the graph.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing graph..."):
                # Run LangGraph workflow
                result = rag_agent_app.invoke({"question": prompt})
                
                answer = result.get("answer", "No answer generated.")
                context = result.get("context", [])
                entities = result.get("entities", [])
                
                st.markdown(answer)
                
                with st.expander("View Retrieved Graph Context"):
                    st.write(f"**Extracted Entities from Query:** {', '.join(entities) if entities else 'None'}")
                    st.write("**Context from Graph:**")
                    if context:
                        for c in context:
                            st.write(f"- {c}")
                    else:
                        st.write("No relevant context found.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "context": context
                })

graph.close()

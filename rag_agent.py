from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from ingestion_agent import get_groq_llm
from graph_utils import Neo4jGraph

# Define the State for LangGraph
class RagState(TypedDict):
    question: str
    entities: List[str]
    context: List[str]
    answer: str

def extract_entities_from_query(state: RagState):
    """Extract key entities from the user's question to query the graph."""
    question = state["question"]
    # Using a smaller/faster model for quick entity extraction
    llm = get_groq_llm(model_name="llama3-8b-8192")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an entity extractor. Extract the core entities (names, organizations, places, key concepts) from the following question. Return ONLY a comma-separated list of the entities, nothing else. If no distinct entities are found, return 'none'."),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    try:
        result = chain.invoke({"question": question})
        content = result.content.strip()
        if content.lower() == 'none' or not content:
            entities = []
        else:
            entities = [e.strip() for e in content.split(',') if e.strip()]
    except Exception as e:
        print(f"Error extracting entities: {e}")
        entities = []
        
    return {"entities": entities}

def retrieve_graph_context(state: RagState):
    """Retrieve 1-hop neighborhood from Neo4j for the extracted entities."""
    entities = state["entities"]
    if not entities:
        return {"context": []}
        
    graph = Neo4jGraph()
    context = graph.get_graph_context(entities)
    graph.close()
    
    return {"context": context}

def generate_answer(state: RagState):
    """Generate the final answer using Groq and the retrieved graph context."""
    question = state["question"]
    context = state["context"]
    
    # Using a larger model for synthesis and generation
    llm = get_groq_llm(model_name="llama3-70b-8192")
    
    context_str = "\n".join(context) if context else "No relevant context found in the graph."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and intelligent AI assistant. Use the following graph context (which represents relationships between entities extracted from a knowledge graph) to answer the user's question. \n\n"
                   "Graph Context:\n{context}\n\n"
                   "Instructions:\n"
                   "- If the context provides the answer, state it clearly.\n"
                   "- If the context is empty or insufficient, you may use your general knowledge, but explicitly mention that the knowledge graph did not contain specific information about this.\n"
                   "- Keep your answer concise and accurate."),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    try:
        result = chain.invoke({"question": question, "context": context_str})
        answer = result.content
    except Exception as e:
        print(f"Error generating answer: {e}")
        answer = "I'm sorry, I encountered an error while trying to generate the answer."
    
    return {"answer": answer}

# Build LangGraph Workflow
workflow = StateGraph(RagState)

workflow.add_node("extract_entities", extract_entities_from_query)
workflow.add_node("retrieve_context", retrieve_graph_context)
workflow.add_node("generate_answer", generate_answer)

workflow.set_entry_point("extract_entities")
workflow.add_edge("extract_entities", "retrieve_context")
workflow.add_edge("retrieve_context", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile the graph
rag_agent_app = workflow.compile()

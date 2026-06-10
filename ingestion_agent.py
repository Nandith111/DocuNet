import os
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define schema for extraction
class Entity(BaseModel):
    id: str = Field(description="Unique identifier for the entity, lowercase and snake_case (e.g., 'elon_musk').")
    label: str = Field(description="The type of the entity (e.g., 'Person', 'Organization', 'Concept').")
    name: str = Field(description="The display name of the entity.")

class Relationship(BaseModel):
    source: str = Field(description="The id of the source entity.")
    target: str = Field(description="The id of the target entity.")
    type: str = Field(description="The type of relationship (e.g., 'FOUNDED', 'WORKS_AT'). Uppercase and snake_case.")

class GraphData(BaseModel):
    entities: List[Entity] = Field(description="List of entities extracted from the text.", default_factory=list)
    relationships: List[Relationship] = Field(description="List of relationships between the extracted entities.", default_factory=list)

def get_groq_llm(model_name="llama3-70b-8192"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError("GROQ_API_KEY is missing or invalid. Please check your .env file.")
    return ChatGroq(model=model_name, temperature=0, api_key=api_key)

def extract_graph_from_text(text: str) -> GraphData:
    """
    Uses Groq LLM to extract structured entities and relationships from a text chunk.
    """
    llm = get_groq_llm()
    # Using with_structured_output to force the LLM to return JSON matching our Pydantic schema
    structured_llm = llm.with_structured_output(GraphData)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert data scientist and knowledge graph extractor. "
                   "Your task is to extract entities and their relationships from the provided text. "
                   "Ensure that the 'source' and 'target' in your relationships perfectly match the 'id' of the entities you extract. "
                   "If you cannot find any distinct entities or relationships, return empty lists."),
        ("human", "Extract graph data from the following text:\n\n{text}")
    ])
    
    chain = prompt | structured_llm
    try:
        result = chain.invoke({"text": text})
        return result
    except Exception as e:
        print(f"Error during extraction: {e}")
        return GraphData(entities=[], relationships=[])

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """A simple naive character chunker. For production, use RecursiveCharacterTextSplitter."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        current_chunk.append(word)
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

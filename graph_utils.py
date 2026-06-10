import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

class Neo4jGraph:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        
        if not uri or not password:
            print("Warning: Neo4j credentials not fully set in .env file.")
            self.driver = None
        else:
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
            except Exception as e:
                print(f"Error connecting to Neo4j: {e}")
                self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def add_document_graph(self, entities, relationships):
        """
        Adds a set of entities and relationships to the graph.
        entities: list of dicts [{'id': 'person_1', 'label': 'Person', 'name': 'John Doe'}]
        relationships: list of dicts [{'source': 'person_1', 'target': 'org_1', 'type': 'WORKS_AT'}]
        """
        if not self.driver:
            return

        with self.driver.session() as session:
            # First, create all entity nodes
            for entity in entities:
                label = entity.get('label', 'Entity').replace(' ', '_').replace('-', '_')
                # Cypher parameters need to be sanitized
                if not label.isalnum():
                    label = "Entity"
                
                query = f"""
                MERGE (e:{label} {{id: $id}})
                SET e.name = $name
                """
                session.run(query, id=entity['id'], name=entity.get('name', entity['id']))
            
            # Then, create relationships
            for rel in relationships:
                rel_type = rel.get('type', 'RELATES_TO').upper().replace(' ', '_').replace('-', '_')
                if not rel_type.isalnum() and '_' not in rel_type:
                    rel_type = "RELATES_TO"
                
                query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                """
                session.run(query, source_id=rel['source'], target_id=rel['target'])

    def get_graph_context(self, entities):
        """
        Given a list of entity names/ids, retrieves a 1-hop neighborhood.
        """
        if not self.driver or not entities:
            return []

        context = []
        with self.driver.session() as session:
            for entity_name in entities:
                query = """
                MATCH (n)-[r]-(m)
                WHERE toLower(n.name) CONTAINS toLower($entity_name) OR toLower(n.id) CONTAINS toLower($entity_name)
                RETURN n.name AS source_name, type(r) AS rel_type, m.name AS target_name
                LIMIT 50
                """
                results = session.run(query, entity_name=entity_name)
                for record in results:
                    context.append(f"{record['source_name']} -[{record['rel_type']}]-> {record['target_name']}")
        
        # Deduplicate
        return list(set(context))

    def get_all_nodes_summary(self):
        """Returns a quick count of nodes to verify ingestion."""
        if not self.driver:
            return {"nodes": 0, "relationships": 0}
            
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            return {"nodes": nodes, "relationships": rels}

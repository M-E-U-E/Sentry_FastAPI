import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from crewai import Agent, Task, Crew
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

VECTOR_IDS_FILE = "vector_ids.json"

class Config:
    """Configuration management class"""
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.github_repo_base = os.getenv("GITHUB_REPO_BASE")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "my-custom-index")
        self.pinecone_endpoint = os.getenv("PINECONE_ENDPOINT")  # Optional endpoint
        self.github_token = os.getenv("GITHUB_TOKEN")
        self._validate_config()

    def _validate_config(self):
        """Validate that all required environment variables are set"""
        required_vars = {
            "GEMINI_API_KEY": self.gemini_api_key,
            "GITHUB_REPO_BASE": self.github_repo_base,
            "PINECONE_API_KEY": self.pinecone_api_key
        }
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class DocumentationManager:
    """
    Manages documentation content and processing.
    All indexed data is stored in Pinecone. The stored metadata for each section includes
    the document name, section index, and the original content.
    """
    def __init__(self, config: Config):
        self.config = config
        self.embedder = SentenceTransformer('paraphrase-mpnet-base-v2')
        # Initialize Pinecone client with endpoint if provided
        self.pc = Pinecone(
            api_key=config.pinecone_api_key,
            environment=config.pinecone_environment,
            endpoint=config.pinecone_endpoint
        )
        try:
            # Check for existing indexes
            existing_indexes = self.pc.list_indexes().names()
            logger.info(f"Existing indexes: {existing_indexes}")

            # Create or connect to index
            if config.pinecone_index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {config.pinecone_index_name}")
                self.pc.create_index(
                    name=config.pinecone_index_name,
                    dimension=768,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='gcp', region='us-east-1')
                )
                logger.info(f"Successfully created index: {config.pinecone_index_name}")
            else:
                logger.info(f"Connected to existing index: {config.pinecone_index_name}")
            
            self.index = self.pc.Index(config.pinecone_index_name)
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize Pinecone: {str(e)}"
            )
        # Load persisted vector IDs if available; otherwise, start with an empty list
        try:
            with open(VECTOR_IDS_FILE, "r", encoding="utf-8") as f:
                self.pinecone_ids: List[str] = json.load(f)
            logger.info(f"Loaded {len(self.pinecone_ids)} vector IDs from {VECTOR_IDS_FILE}")
        except FileNotFoundError:
            self.pinecone_ids = []

    async def fetch_markdown_files(self, folder_path: str = "") -> Dict[str, str]:
        """
        Fetches Markdown files from the GitHub repository.
        """
        docs_content = {}
        current_repo_url = (f"{self.config.github_repo_base}/contents/{folder_path}" 
                            if folder_path else f"{self.config.github_repo_base}/contents")
        headers = {"Authorization": f"Bearer {self.config.github_token}"} if self.config.github_token else {}
        try:
            response = requests.get(current_repo_url, headers=headers)
            response.raise_for_status()
            files = response.json()
            for file_info in files:
                if file_info['type'] == 'dir':
                    new_folder_path = os.path.join(folder_path, file_info['name'])
                    docs_content.update(await self.fetch_markdown_files(new_folder_path))
                elif file_info['name'].endswith(".md"):
                    file_response = requests.get(file_info['download_url'], headers=headers)
                    file_response.raise_for_status()
                    docs_content[file_info['name']] = file_response.text
            return docs_content
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching markdown files: {str(e)}")
            raise HTTPException(
                status_code=e.response.status_code if hasattr(e, 'response') else 500,
                detail=str(e)
            )

    def process_documentation(self, docs_content: Dict[str, str]):
        """
        Process and index documentation content.
        Each non-empty section from a Markdown file is embedded and stored in Pinecone.
        The metadata for each vector now includes the section’s original text.
        """
        try:
            # Clear any previous IDs since we are processing new documentation
            self.pinecone_ids = []
            batch_size = 100
            vectors_to_upsert = []
            
            for doc_name, content in docs_content.items():
                doc_sections = content.split('\n\n')
                for i, section in enumerate(doc_sections):
                    if section.strip():
                        vector_id = f"{doc_name}-{i}"
                        try:
                            embedding = self.embedder.encode(section)
                            vector = {
                                'id': vector_id,
                                'values': embedding.tolist(),
                                'metadata': {
                                    'doc_name': doc_name,
                                    'section_index': i,
                                    'content': section
                                }
                            }
                            vectors_to_upsert.append(vector)
                            self.pinecone_ids.append(vector_id)
                            if len(vectors_to_upsert) >= batch_size:
                                self.index.upsert(vectors=vectors_to_upsert)
                                vectors_to_upsert = []
                        except Exception as e:
                            logger.error(f"Error processing section {i} of {doc_name}: {str(e)}")
                            continue
                if vectors_to_upsert:
                    self.index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
            # Persist the vector IDs to a file
            with open(VECTOR_IDS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.pinecone_ids, f)
            logger.info(f"Indexed {len(self.pinecone_ids)} sections into Pinecone.")
        except Exception as e:
            logger.error(f"Error processing documentation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from Pinecone using the user's query.
        """
        try:
            # Create embedding for the query
            query_embedding = self.embedder.encode(query)
            # Query Pinecone for the most relevant documents
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            matching_docs = [
                {
                    "doc_name": match["metadata"]["doc_name"],
                    "section_index": match["metadata"]["section_index"],
                    "content": match["metadata"]["content"]
                }
                for match in results.get("matches", [])
            ]
            return matching_docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

class QueryInput(BaseModel):
    """Input model for documentation queries"""
    query: str = Field(..., min_length=1, description="The user's query")
    user_context: str = Field(default="", description="Additional context for the query")

class CrewManager:
    """Manages CrewAI agents and tasks"""
    def __init__(self, config: Config):
        self.config = config
        self.setup_agents()
        self.setup_tasks()

    def setup_agents(self):
        """Initialize CrewAI agents"""
        self.agents = {
            'retriever': Agent(
                role="Documentation Retriever",
                goal="Identify and fetch relevant documentation sections based on user queries.",
                backstory="A specialized AI designed to efficiently search documentation using embeddings.",
                tools=[],
                verbose=True,
                memory=True,
                llm="gemini/gemini-1.5-flash-latest"
            ),
            'analyzer': Agent(
                role="Content Analyzer",
                goal="Process retrieved documentation to extract structured insights.",
                backstory="An AI that specializes in organizing and structuring knowledge for easy consumption.",
                tools=[],
                verbose=True,
                memory=True,
                llm="gemini/gemini-1.5-pro-latest"
            ),
            'assistant': Agent(
                role="Documentation Guide",
                goal="Provide clear and well-structured answers based on relevant documentation.",
                backstory="A helpful AI that presents technical documentation in an easy-to-understand way.",
                verbose=True,
                memory=True,
                llm="gemini/gemini-1.5-pro-latest"
            )
        }

    def setup_tasks(self):
        """Initialize tasks"""
        self.tasks = [
            Task(
                description="Retrieve the most relevant documentation sections for a given query using vector search.",
                expected_output="A list of relevant documentation sections.",
                agent=self.agents['retriever']
            ),
            Task(
                description="Analyze and summarize the retrieved documentation sections, extracting key points.",
                expected_output="A structured summary of key insights from the retrieved sections.",
                agent=self.agents['analyzer']
            ),
            Task(
                description="Generate a well-structured response for the user, incorporating retrieved insights.",
                expected_output="A detailed response with references to specific documentation sections.",
                agent=self.agents['assistant']
            )
        ]

    def create_query_tasks(self, query: str, user_context: str, matching_docs: Dict[str, Any]) -> List[Task]:
        """Create specialized tasks for a specific query"""
        return [
            Task(
                description=f"""
                    Identify and fetch the most relevant documentation sections based on the user's query.
                    User Query: {query}
                    User Context: {user_context}
                """,
                expected_output="A list of relevant documentation sections with metadata.",
                agent=self.agents['retriever']
            ),
            Task(
                description=f"""
                    Analyze the retrieved documentation sections and summarize the key insights.
                    Query: {query}
                    Relevant Documentation: {json.dumps(matching_docs)}
                """,
                expected_output="A well-structured summary of key findings from the retrieved sections.",
                agent=self.agents['analyzer']
            ),
            Task(
                description=f"""
                    Formulate a detailed and structured response to the user's query, referencing relevant documentation.
                    Ensure the answer is easy to understand and includes direct citations.
                    Query: {query}
                    Summary from Analysis: [Include key points]
                """,
                expected_output="A clear, structured response to the user’s query with references.",
                agent=self.agents['assistant']
            )
        ]


# --- Create singleton instances so state persists across requests ---
_config_instance = Config()
_doc_manager_instance = DocumentationManager(_config_instance)
_crew_manager_instance = CrewManager(_config_instance)

# Initialize FastAPI app
app = FastAPI(title="Documentation Assistant API")

# Dependency injection functions now return the singleton instances
async def get_config() -> Config:
    return _config_instance

async def get_doc_manager(config: Config = Depends(get_config)) -> DocumentationManager:
    return _doc_manager_instance

async def get_crew_manager(config: Config = Depends(get_config)) -> CrewManager:
    return _crew_manager_instance

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Documentation Assistant API",
        "version": "2.0.0"
    }

@app.get("/fetch_docs")
async def fetch_docs(
    doc_manager: DocumentationManager = Depends(get_doc_manager)
):
    """
    Fetch and process documentation asynchronously:
      - Retrieve Markdown files from GitHub.
      - Process content into sections.
      - Generate embeddings and store them in Pinecone.
    """
    try:
        logger.info("Fetching documentation from GitHub...")
        docs_content = await doc_manager.fetch_markdown_files()
        
        if not docs_content:
            raise HTTPException(status_code=404, detail="No documentation found in the repository.")
        
        logger.info(f"Processing {len(docs_content)} documentation files...")
        doc_manager.process_documentation(docs_content)

        return {
            "message": "Documentation successfully fetched and indexed.",
            "stats": {
                "docs_count": len(docs_content),
                "indexed_sections_count": len(doc_manager.pinecone_ids)
            }
        }
    except Exception as e:
        logger.error(f"Error fetching documentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_doc_assistant")
async def ask_doc_assistant(
    query_input: QueryInput,
    doc_manager: DocumentationManager = Depends(get_doc_manager),
    crew_manager: CrewManager = Depends(get_crew_manager)
):
    """
    Process user query using RAG:
    1. Retrieve relevant documentation sections via Pinecone.
    2. Generate structured responses based on relevant sections.
    3. Return a well-formatted response.
    """
    try:
        logger.info(f"Processing query: {query_input.query}")

        # Step 1: Retrieve relevant documentation
        matching_docs = doc_manager.retrieve_documents(query_input.query)
        if not matching_docs:
            return {"response": "No relevant documentation found.", "matching_docs": []}

        # Step 2: Create CrewAI tasks for query handling
        query_tasks = crew_manager.create_query_tasks(query_input.query, query_input.user_context, matching_docs)

        # Step 3: Execute tasks via CrewAI
        crew = Crew(
            agents=[crew_manager.agents['retriever'], crew_manager.agents['analyzer'], crew_manager.agents['assistant']],
            tasks=query_tasks,
            verbose=True
        )
        result = crew.kickoff()

        # Step 4: Format response
        structured_response = {
            "response": result,
            "query_context": {
                "original_query": query_input.query,
                "user_context": query_input.user_context
            },
            "matching_docs": matching_docs
        }

        logger.info("Query processed successfully.")
        return structured_response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

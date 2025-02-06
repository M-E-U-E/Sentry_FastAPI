import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import requests
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, HttpUrl, validator
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
MAX_RETRIES = 3
BATCH_SIZE = 100

class ConfigModel(BaseModel):
    """Pydantic model for configuration validation"""
    gemini_api_key: str
    github_repo_base: HttpUrl
    pinecone_api_key: str
    pinecone_environment: str = Field(default="us-east-1")
    pinecone_index_name: str = Field(default="my-custom-index")
    pinecone_endpoint: Optional[HttpUrl] = None
    github_token: Optional[str] = None
    max_workers: int = Field(default=4, ge=1, le=10)

class Config:
    """Configuration management class with validation"""
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.github_repo_base = os.getenv("GITHUB_REPO_BASE")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "my-custom-index")
        self.pinecone_endpoint = os.getenv("PINECONE_ENDPOINT")
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        self._validate_config()
        
        # Validate using Pydantic model
        self.config_model = ConfigModel(
            gemini_api_key=self.gemini_api_key,
            github_repo_base=self.github_repo_base,
            pinecone_api_key=self.pinecone_api_key,
            pinecone_environment=self.pinecone_environment,
            pinecone_index_name=self.pinecone_index_name,
            pinecone_endpoint=self.pinecone_endpoint,
            github_token=self.github_token,
            max_workers=self.max_workers
        )

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

class DocumentSection(BaseModel):
    """Validation model for document sections"""
    content: str
    doc_name: str
    section_index: int

    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()

class DocumentationManager:
    """
    Enhanced documentation manager with parallel processing and validation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.embedder = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.last_processed_timestamp = datetime.now()
        self._init_pinecone()
        self._load_vector_ids()

    def _init_pinecone(self) -> None:
        """Initialize Pinecone with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                self.pc = Pinecone(
                    api_key=self.config.pinecone_api_key,
                    environment=self.config.pinecone_environment,
                    endpoint=self.config.pinecone_endpoint
                )
                existing_indexes = self.pc.list_indexes().names()
                logger.info(f"Existing indexes: {existing_indexes}")

                if self.config.pinecone_index_name not in existing_indexes:
                    logger.info(f"Creating new Pinecone index: {self.config.pinecone_index_name}")
                    self.pc.create_index(
                        name=self.config.pinecone_index_name,
                        dimension=768,
                        metric='cosine',
                        spec=ServerlessSpec(cloud='gcp', region='us-east-1')
                    )
                
                self.index = self.pc.Index(self.config.pinecone_index_name)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to initialize Pinecone after {MAX_RETRIES} attempts: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Pinecone initialization failed: {str(e)}")
                logger.warning(f"Retrying Pinecone initialization. Attempt {attempt + 1}/{MAX_RETRIES}")
                asyncio.sleep(1)

    def _load_vector_ids(self) -> None:
        """Load persisted vector IDs with error handling"""
        try:
            if os.path.exists(VECTOR_IDS_FILE):
                with open(VECTOR_IDS_FILE, "r", encoding="utf-8") as f:
                    self.pinecone_ids = json.load(f)
                logger.info(f"Loaded {len(self.pinecone_ids)} vector IDs")
            else:
                self.pinecone_ids = []
        except json.JSONDecodeError as e:
            logger.error(f"Error loading vector IDs: {str(e)}")
            self.pinecone_ids = []

    async def fetch_markdown_files(self, folder_path: str = "") -> Dict[str, str]:
        """
        Asynchronously fetch Markdown files using aiohttp
        """
        async def fetch_file(session: aiohttp.ClientSession, url: str, filename: str) -> tuple[str, str]:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.text()
                return filename, content

        docs_content = {}
        headers = {"Authorization": f"Bearer {self.config.github_token}"} if self.config.github_token else {}
        
        async with aiohttp.ClientSession(headers=headers) as session:
            try:
                current_repo_url = (f"{self.config.github_repo_base}/contents/{folder_path}" 
                                  if folder_path else f"{self.config.github_repo_base}/contents")
                
                async with session.get(current_repo_url) as response:
                    response.raise_for_status()
                    files = await response.json()

                tasks = []
                for file_info in files:
                    if file_info['type'] == 'dir':
                        new_folder_path = os.path.join(folder_path, file_info['name'])
                        nested_content = await self.fetch_markdown_files(new_folder_path)
                        docs_content.update(nested_content)
                    elif file_info['name'].endswith('.md'):
                        tasks.append(
                            fetch_file(session, file_info['download_url'], file_info['name'])
                        )

                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching file: {str(result)}")
                    else:
                        filename, content = result
                        docs_content[filename] = content

                return docs_content

            except aiohttp.ClientError as e:
                logger.error(f"Error fetching markdown files: {str(e)}")
                raise HTTPException(
                    status_code=e.response.status_code if hasattr(e, 'response') else 500,
                    detail=str(e)
                )

    def _create_document_section(self, doc_name: str, section: str, index: int) -> Optional[DocumentSection]:
        """Create and validate a document section with better content cleaning"""
        # Clean the section content more thoroughly
        cleaned_content = ' '.join(section.split())  # Remove excessive whitespace
        
        if not cleaned_content:  # Skip if content is empty after cleaning
            return None
            
        try:
            return DocumentSection(
                content=cleaned_content,
                doc_name=doc_name,
                section_index=index
            )
        except ValueError as e:
            logger.debug(f"Skipping invalid section in {doc_name} at index {index}: {str(e)}")
            return None

    def _embed_section(self, section: DocumentSection) -> dict:
        """Create embedding for a document section"""
        embedding = self.embedder.encode(section.content)
        return {
            'id': f"{section.doc_name}-{section.section_index}",
            'values': embedding.tolist(),
            'metadata': {
                'doc_name': section.doc_name,
                'section_index': section.section_index,
                'content': section.content,
                'timestamp': self.last_processed_timestamp.isoformat()
            }
        }

    def _upsert_batch(self, vectors: List[dict]) -> None:
        """Upsert a batch of vectors with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                self.index.upsert(vectors=vectors)
                return
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to upsert batch after {MAX_RETRIES} attempts: {str(e)}")
                    raise
                logger.warning(f"Retrying batch upsert. Attempt {attempt + 1}/{MAX_RETRIES}")
                asyncio.sleep(1)

    def process_documentation(self, docs_content: Dict[str, str]) -> None:
        """
        Process documentation with improved content splitting and validation
        """
        try:
            self.last_processed_timestamp = datetime.now()
            self.pinecone_ids = []
            sections_to_process = []

            # Improved content splitting with better section handling
            for doc_name, content in docs_content.items():
                # Split content more intelligently
                doc_sections = [
                    section for section in content.split('\n\n')
                    if section.strip()  # Pre-filter empty sections
                ]
                
                meaningful_sections = []
                current_section = []
                
                for section in doc_sections:
                    cleaned_section = section.strip()
                    if cleaned_section:
                        current_section.append(cleaned_section)
                        # If section is substantial enough, add it
                        if len(' '.join(current_section)) >= 50:  # Minimum content length
                            combined_section = ' '.join(current_section)
                            if doc_section := self._create_document_section(
                                doc_name, 
                                combined_section, 
                                len(meaningful_sections)
                            ):
                                meaningful_sections.append(doc_section)
                                current_section = []
                
                # Add any remaining content
                if current_section:
                    combined_section = ' '.join(current_section)
                    if doc_section := self._create_document_section(
                        doc_name, 
                        combined_section, 
                        len(meaningful_sections)
                    ):
                        meaningful_sections.append(doc_section)
                
                sections_to_process.extend(meaningful_sections)

            # Process sections in parallel
            total_sections = len(sections_to_process)
            logger.info(f"Processing {total_sections} meaningful sections...")

            if not sections_to_process:
                logger.warning("No valid sections found to process")
                return

            # Process sections in parallel with progress tracking
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_section = {
                    executor.submit(self._embed_section, section): section 
                    for section in sections_to_process
                }

                vectors_to_upsert = []
                completed = 0

                for future in as_completed(future_to_section):
                    try:
                        vector = future.result()
                        vectors_to_upsert.append(vector)
                        self.pinecone_ids.append(vector['id'])

                        if len(vectors_to_upsert) >= BATCH_SIZE:
                            self._upsert_batch(vectors_to_upsert)
                            vectors_to_upsert = []

                        completed += 1
                        if completed % 10 == 0:  # Log progress every 10 sections
                            logger.info(f"Processed {completed}/{total_sections} sections")

                    except Exception as e:
                        section = future_to_section[future]
                        logger.error(f"Error processing section {section.section_index} "
                                f"of {section.doc_name}: {str(e)}")

                # Upsert any remaining vectors
                if vectors_to_upsert:
                    self._upsert_batch(vectors_to_upsert)

            # Persist vector IDs
            with open(VECTOR_IDS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.pinecone_ids, f)

            logger.info(f"Successfully processed {len(self.pinecone_ids)} sections")

        except Exception as e:
            logger.error(f"Error in process_documentation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def retrieve_documents(self, query: str, top_k: int = 5, min_score: float = 0.7) -> List[Dict[str, Any]]:
        try:
            # Generate embedding for the query
            query_embedding = self.embedder.encode(query).tolist()

            # Query the Pinecone index
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Filter results based on similarity score
            matching_docs = []
            for match in response['matches']:
                if match['score'] >= min_score:
                    matching_docs.append({
                        "doc_name": match['metadata']['doc_name'],
                        "similarity_score": match['score'],
                        "processed_at": match['metadata'].get('timestamp', ''),
                        "content": match['metadata'].get('content', "")
                    })

            return matching_docs

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")


class QueryInput(BaseModel):
    """Validation model for query inputs"""
    query: str = Field(..., min_length=3, max_length=1000)
    user_context: Optional[str] = Field(default="", max_length=500)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I setup the application?",
                "user_context": "I'm a new developer trying to understand the system."
            }
        }

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
                llm="gemini/gemini-2.0-flash"
            ),
            'analyzer': Agent(
                role="Content Analyzer",
                goal="Process retrieved documentation to extract structured insights.",
                backstory="An AI that specializes in organizing and structuring knowledge for easy consumption.",
                tools=[],
                verbose=True,
                memory=True,
                llm="gemini/gemini-2.0-flash"
            ),
            'assistant': Agent(
                role="Documentation Guide",
                goal="Provide clear and well-structured answers based on relevant documentation.",
                backstory="A helpful AI that presents technical documentation in an easy-to-understand way.",
                verbose=True,
                memory=True,
                llm="gemini/gemini-2.0-flash"
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

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Documentation Assistant API",
    description="AI-powered documentation assistant with RAG capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced dependency injection
async def get_config() -> Config:
    return _config_instance

async def get_doc_manager(config: Config = Depends(get_config)) -> DocumentationManager:
    return _doc_manager_instance

async def get_crew_manager(config: Config = Depends(get_config)) -> CrewManager:
    return _crew_manager_instance

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Documentation Assistant API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/fetch_docs", tags=["Documentation"])
async def fetch_docs(
    doc_manager: DocumentationManager = Depends(get_doc_manager)
):
    """
    Fetch and process documentation asynchronously with enhanced error handling
    and progress tracking.
    """
    try:
        logger.info("Starting documentation fetch process...")
        docs_content = await doc_manager.fetch_markdown_files()
        
        if not docs_content:
            raise HTTPException(
                status_code=404, 
                detail="No documentation found in the repository"
            )
        
        logger.info(f"Processing {len(docs_content)} documentation files...")
        doc_manager.process_documentation(docs_content)
        
        # Enhanced response with detailed statistics
        return {
            "status": "success",
            "message": "Documentation successfully fetched and indexed",
            "stats": {
                "files_processed": len(docs_content),
                "sections_indexed": len(doc_manager.pinecone_ids),
                "processed_at": datetime.now().isoformat(),
                "files": list(docs_content.keys())
            }
        }
    except Exception as e:
        logger.error(f"Error in documentation fetch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Documentation fetch failed: {str(e)}"
        )

@app.post("/ask_doc_assistant", tags=["Query"])
async def ask_doc_assistant(
    query_input: QueryInput,
    doc_manager: DocumentationManager = Depends(get_doc_manager),
    crew_manager: CrewManager = Depends(get_crew_manager)
):
    """
    Process user queries using RAG with enhanced error handling and response structure.
    """
    try:
        logger.info(f"Processing query: {query_input.query}")
        query_start_time = datetime.now()

        # Retrieve relevant documentation with similarity threshold
        matching_docs = await doc_manager.retrieve_documents(
            query=query_input.query,
            top_k=15,
            min_score=0.35
        )

        if not matching_docs:
            return {
                "status": "no_results",
                "response": "No relevant documentation found for your query.",
                "query_context": {
                    "original_query": query_input.query,
                    "user_context": query_input.user_context
                },
                "matching_docs": []
            }

        # Create and execute CrewAI tasks
        query_tasks = crew_manager.create_query_tasks(
            query_input.query,
            query_input.user_context,
            matching_docs
        )

        crew = Crew(
            agents=[
                crew_manager.agents['retriever'],
                crew_manager.agents['analyzer'],
                crew_manager.agents['assistant']
            ],
            tasks=query_tasks,
            verbose=True
        )
        
        result = crew.kickoff()

        # Enhanced response structure
        structured_response = {
            "status": "success",
            "response": result,
            "query_context": {
                "original_query": query_input.query,
                "user_context": query_input.user_context,
                "processing_time": (datetime.now() - query_start_time).total_seconds()
            },
            "matching_docs": [
                {
                    "doc_name": doc["doc_name"],
                    "similarity_score": doc["similarity_score"],
                    "processed_at": doc["processed_at"],
                    "content_preview": doc["content"][:200] + "..."
                }
                for doc in matching_docs
            ]
        }

        logger.info("Query processed successfully")
        return structured_response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "message": str(e),
                "query": query_input.query
            }
        )

# Health check endpoint
@app.get("/health", tags=["General"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "pinecone": "operational",
            "documentation_manager": "operational",
            "crew_manager": "operational"
        }
    }
@app.delete("/clear_index", tags=["Maintenance"])
async def clear_pinecone_index(
    doc_manager: DocumentationManager = Depends(get_doc_manager)
):
    """
    Clear all vectors from the Pinecone index.
    """
    try:
        logger.info("Deleting all vectors from Pinecone index...")
        doc_manager.index.delete(delete_all=True)

        # Clear the local vector ID list as well
        doc_manager.pinecone_ids = []
        with open(VECTOR_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

        return {
            "status": "success",
            "message": "All vectors deleted from the Pinecone index."
        }
    except Exception as e:
        logger.error(f"Error clearing Pinecone index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Pinecone index: {str(e)}")
import os
import json
import logging
import sentry_sdk
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import aiohttp
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel, Field, HttpUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()



VECTOR_IDS_FILE = "vector_ids.json"
MAX_RETRIES = 3
BATCH_SIZE = 100


# Sentry Configuration

class SentryConfig:
    """
    Class to encapsulate Sentry SDK initialization and event handling.
    """
    @classmethod
    def init_sentry(cls):
        """
        Initializes Sentry with environment-based configuration.
        """
        dsn = os.getenv("SENTRY_DSN")
        if not dsn:
            logger.warning("SENTRY_DSN not found in environment. Sentry will not be enabled.")
            return

        sentry_sdk.init(
            dsn=dsn,
            debug=os.getenv("SENTRY_DEBUG", "false").lower() == "true",
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            traces_sample_rate=float(os.getenv("SENTRY_SAMPLE_RATE", "1.0")),
            profiles_sample_rate=float(os.getenv("SENTRY_ERROR_SAMPLE_RATE", "1.0")),
            send_default_pii=True,
            attach_stacktrace=True,
            include_source_context=True,
            include_local_variables=True,
            max_breadcrumbs=50,
            server_name=os.getenv("SERVER_NAME", "fastapi-server"),
            before_send=cls.before_send,
        )
        logger.info("Sentry successfully initialized.")

    @staticmethod
    def before_send(event, hint):
        """
        Scrubs sensitive information before sending an event to Sentry.
        Can be customized to exclude certain errors.
        """
        if "exc_info" in hint:
            exc_type, exc_value, _ = hint["exc_info"]
            if isinstance(exc_value, HTTPException) and exc_value.status_code in [400, 404]:  # Ignore common errors
                return None  
        return event





class ConfigModel(BaseModel):
    gemini_api_key: str
    github_repo_base: HttpUrl
    pinecone_api_key: str
    pinecone_environment: str = Field(default="us-east-1")
    pinecone_index_name: str = Field(default="my-custom-index")
    pinecone_endpoint: Optional[HttpUrl] = None
    github_token: Optional[str] = None
    max_workers: int = Field(default=4, ge=1, le=10)

class Config:
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
        required_vars = {
            "GEMINI_API_KEY": self.gemini_api_key,
            "GITHUB_REPO_BASE": self.github_repo_base,
            "PINECONE_API_KEY": self.pinecone_api_key
        }
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

class DocumentSection(BaseModel):
    content: str
    doc_name: str
    section_index: int

class DocumentationManager:
    def __init__(self, config: Config):
        self.config = config
        self.embedder = SentenceTransformer('paraphrase-mpnet-base-v2')
        self.last_processed_timestamp = datetime.now()
        self._init_pinecone()
        self._load_vector_ids()

    def _init_pinecone(self) -> None:
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
        try:
            if os.path.exists(VECTOR_IDS_FILE):
                with open(VECTOR_IDS_FILE, "r", encoding="utf-8") as f:
                    self.pinecone_ids = json.load(f)
                logger.info(f"Loaded {len(self.pinecone_ids)} vector IDs")
            else:
                self.pinecone_ids = []
        except Exception as e:
            logger.error(f"Error loading vector IDs: {str(e)}")
            self.pinecone_ids = []

    async def fetch_markdown_files(self, folder_path: str = "") -> Dict[str, str]:
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

    def _embed_section(self, section: DocumentSection) -> dict:
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
        try:
            self.last_processed_timestamp = datetime.now()
            self.pinecone_ids = []
            vectors_to_upsert = []

            for doc_name, content in docs_content.items():
                # Store complete document content
                doc_section = DocumentSection(
                    content=content,
                    doc_name=doc_name,
                    section_index=0
                )

                vector = self._embed_section(doc_section)
                vectors_to_upsert.append(vector)
                self.pinecone_ids.append(vector['id'])

                if len(vectors_to_upsert) >= BATCH_SIZE:
                    self._upsert_batch(vectors_to_upsert)
                    vectors_to_upsert = []

            if vectors_to_upsert:
                self._upsert_batch(vectors_to_upsert)

            with open(VECTOR_IDS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.pinecone_ids, f)

            logger.info(f"Successfully processed {len(self.pinecone_ids)} documents")

        except Exception as e:
            logger.error(f"Error in process_documentation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedder.encode(query).tolist()

            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            matching_docs = []
            for match in response['matches']:
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

# Initialize FastAPI app
app = FastAPI(
    title="Documentation Assistant API",
    description="AI-powered documentation assistant with complete content preservation",
    version="2.0.0"
)

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
                expected_output="A clear, structured response to the user's query with references.",
                agent=self.agents['assistant']
            )
        ]
        # Dependency injection functions
def get_config() -> Config:
    return _config_instance

def get_doc_manager(config: Config = Depends(get_config)) -> DocumentationManager:
    return _doc_manager_instance

def get_crew_manager(config: Config = Depends(get_config)) -> CrewManager:
    return _crew_manager_instance

# Add CrewManager dependency
async def get_crew_manager(config: Config = Depends(get_config)) -> CrewManager:
    return _crew_manager_instance

# Initialize singleton instances
_config_instance = Config()
_doc_manager_instance = DocumentationManager(_config_instance)
_crew_manager_instance = CrewManager(_config_instance)

# Initialize FastAPI app
app = FastAPI(
    title="Documentation Assistant API",
    description="AI-powered documentation assistant with complete content preservation",
    version="2.0.0"
)
# Initialize Sentry
SentryConfig.init_sentry()



# Custom error-handling decorator for API routes
def sentry_error_handler(func):
    """
    Decorator to wrap FastAPI route handlers and send errors to Sentry.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    return wrapper

# Add CrewAI endpoint

@app.get("/", tags=["General"])
async def root():
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
    try:
        logger.info("Starting documentation fetch process...")
        docs_content = await doc_manager.fetch_markdown_files()
        
        if not docs_content:
            raise HTTPException(status_code=404, detail="No documentation found")
        
        doc_manager.process_documentation(docs_content)
        
        return {
            "status": "success",
            "message": "Documentation fetched and indexed",
            "stats": {
                "files_processed": len(docs_content),
                "files": list(docs_content.keys()),
                "processed_at": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error in documentation fetch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Documentation fetch failed: {str(e)}")

@app.post("/ask_doc_assistant", tags=["Query"])
async def ask_doc_assistant(
    query_input: QueryInput,
    doc_manager: DocumentationManager = Depends(get_doc_manager),
    crew_manager: CrewManager = Depends(get_crew_manager)
):
    """
    Process user queries using RAG with CrewAI-powered analysis.
    """
    try:
        logger.info(f"Processing query: {query_input.query}")
        query_start_time = datetime.now()

        # Retrieve relevant documentation
        matching_docs = await doc_manager.retrieve_documents(
            query=query_input.query,
            top_k=15
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
                    "content": doc["content"]  # Return full content
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

@app.delete("/clear_index", tags=["Maintenance"])
async def clear_index(
    doc_manager: DocumentationManager = Depends(get_doc_manager)
):
    try:
        logger.info("Clearing Pinecone index...")
        doc_manager.index.delete(delete_all=True)
        doc_manager.pinecone_ids = []
        with open(VECTOR_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        return {"status": "success", "message": "Index cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentry-debug", tags=["Sentry"])
async def trigger_error():
    division_by_zero = 1 / 0


SENTRY_AUTH_TOKEN = os.getenv("SENTRY_AUTH_TOKEN")
ORG_SLUG = os.getenv("ORG_SLUG")
PROJECT_SLUG = os.getenv("PROJECT_SLUG")
GITHUB_WEBHOOK_URL = os.getenv("GITHUB_WEBHOOK_URL")
GITHUB_PAT = os.getenv("GITHUB_PAT")

# Processed issues (to avoid duplicate branches)
processed_issues = set()

class SentryAPI:
    def __init__(self, auth_token: str, organization_slug: str, project_slug: str):
        self.auth_token = auth_token
        self.base_url = "https://sentry.io/api/0"
        self.org_slug = organization_slug
        self.project_slug = project_slug
        self.headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Helper method for making GET requests to Sentry API."""
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from Sentry")
        return response.json()

    def get_latest_issues(self, limit: int = 100) -> List[Dict]:
        """Fetch the latest unresolved issues from Sentry."""
        endpoint = f"/projects/{self.org_slug}/{self.project_slug}/issues/"
        params = {'limit': limit, 'query': 'is:unresolved'}
        return self._get(endpoint, params)

    def get_error_location(self, event_id: str) -> Dict:
        """Get the specific error location including file, line number, and code context."""
        endpoint = f"/projects/{self.org_slug}/{self.project_slug}/events/{event_id}/"
        event_data = self._get(endpoint)

        error_location = {
            'file': None,
            'line_number': None,
            'context_lines': [],
            'error_line': None,
            'function': None,
            'pre_context': [],
            'post_context': []
        }

        if 'entries' in event_data:
            for entry in event_data.get('entries', []):
                if entry.get('type') == 'exception':
                    frames = entry.get('data', {}).get('values', [])[0].get('stacktrace', {}).get('frames', [])
                    if frames:
                        last_frame = frames[-1]
                        error_location.update({
                            'file': last_frame.get('filename'),
                            'line_number': last_frame.get('lineno'),
                            'context_lines': last_frame.get('context_line'),
                            'error_line': last_frame.get('context_line'),
                            'function': last_frame.get('function'),
                            'pre_context': last_frame.get('pre_context', []),
                            'post_context': last_frame.get('post_context', [])
                        })
        return error_location

    def get_full_error_details(self, issue_id: str) -> Dict:
        """Get comprehensive error details including stack trace and error location."""
        endpoint = f"/issues/{issue_id}/events/latest/"
        event_data = self._get(endpoint)

        error_details = {
            'error_type': None,
            'error_message': None,
            'error_location': None,
            'stack_trace': [],
            'timestamp': None
        }

        if 'entries' in event_data:
            for entry in event_data.get('entries', []):
                if entry.get('type') == 'exception':
                    exception = entry.get('data', {}).get('values', [])[0]
                    error_details.update({
                        'error_type': exception.get('type'),
                        'error_message': exception.get('value'),
                        'timestamp': event_data.get('dateCreated')
                    })
                    frames = exception.get('stacktrace', {}).get('frames', [])
                    error_details['stack_trace'] = [
                        {
                            'filename': frame.get('filename'),
                            'function': frame.get('function'),
                            'line_number': frame.get('lineno'),
                            'context_line': frame.get('context_line')
                        }
                        for frame in frames
                    ]
        error_details['error_location'] = self.get_error_location(event_data.get('eventID'))
        return error_details


class ErrorDetailResponse(BaseModel):
    error_type: Optional[str]
    error_message: Optional[str]
    error_location: Optional[Dict]
    stack_trace: List[Dict]
    timestamp: Optional[str]


# Initialize Sentry API
sentry_api = SentryAPI(auth_token=SENTRY_AUTH_TOKEN, organization_slug=ORG_SLUG, project_slug=PROJECT_SLUG)


async def check_sentry_issues():
    """Periodically checks for new Sentry issues and creates GitHub branches."""
    while True:
        try:
            issues = sentry_api.get_latest_issues(limit=100)
            async with httpx.AsyncClient() as client:
                for issue in issues:
                    issue_id = issue["id"]

                    if issue_id in processed_issues:
                        continue  # Skip if the issue was already processed

                    title = issue["title"]
                    permalink = issue["permalink"]
                    error_details = sentry_api.get_full_error_details(issue_id)

                    # Prepare payload for GitHub repository dispatch event
                    webhook_payload = {
                        "event_type": "create-branch",
                        "client_payload": {
                            "branch_name": issue_id,  # Branch name will be the Sentry issue ID
                            "issue_title": title,
                            "permalink": permalink
                        }
                    }

                    # Send the POST request to trigger the GitHub Action
                    headers = {
                        "Authorization": f"token {GITHUB_PAT}",
                        "Accept": "application/vnd.github.v3+json"
                    }
                    webhook_response = await client.post(GITHUB_WEBHOOK_URL, json=webhook_payload, headers=headers)

                    if webhook_response.status_code == 204:
                        print(f"✅ Branch creation triggered for issue {issue_id}")
                        processed_issues.add(issue_id)  # Mark issue as processed
                    else:
                        print(f"❌ Webhook call failed for issue {issue_id}: {webhook_response.status_code} {webhook_response.text}")

        except Exception as e:
            print(f"⚠️ Error fetching Sentry issues: {str(e)}")

        await asyncio.sleep(60)  # Wait 60 seconds before checking again


@app.on_event("startup")
async def startup_event():
    """Start checking for Sentry issues when FastAPI starts."""
    asyncio.create_task(check_sentry_issues())


@app.get("/status")
async def get_status():
    """Check how many issues have been processed."""
    return {"processed_issues": list(processed_issues)}


@app.get("/get-sentry-issues")
async def get_sentry_issues(limit: int = 100):
    issues = sentry_api.get_latest_issues(limit=limit)
    formatted_issues = []
    
    # Use an asynchronous HTTP client to send webhook calls
    async with httpx.AsyncClient() as client:
        for issue in issues:
            issue_id = issue["id"]
            title = issue["title"]
            permalink = issue["permalink"]
            error_details = sentry_api.get_full_error_details(issue_id)

            formatted_issue = {
                "id": issue_id,
                "title": title,
                "permalink": permalink,
                "error_type": error_details['error_type'],
                "error_message": error_details['error_message'],
                "error_location": error_details['error_location'],
                "timestamp": error_details['timestamp']
            }
            formatted_issues.append(formatted_issue)

            # Prepare payload for GitHub repository dispatch event
            webhook_payload = {
                "event_type": "create-branch",
                "client_payload": {
                    "branch_name": issue_id,  # The branch name will be the Sentry issue ID
                    "issue_title": title,
                    "permalink": permalink
                }
            }
            # Send the POST request to trigger the GitHub Action
            headers = {
                "Authorization": f"token {GITHUB_PAT}",
                "Accept": "application/vnd.github.v3+json"
            }
            webhook_response = await client.post(GITHUB_WEBHOOK_URL, json=webhook_payload, headers=headers)
            if webhook_response.status_code != 204:
                # GitHub API returns 204 on success for repository_dispatch events
                print(f"Webhook call failed for issue {issue_id}: {webhook_response.status_code} {webhook_response.text}")

    return {"status": "success", "issues": formatted_issues}


@app.get("/get-sentry-error-details/{issue_id}")
async def get_error_details(issue_id: str):
    """Fetches detailed error information for a specific Sentry issue in the same format as `/get-sentry-issues`."""
    
    # Fetch error details from Sentry API
    error_details = sentry_api.get_full_error_details(issue_id)
    
    # If issue does not exist, return 404
    if not error_details.get("error_type"):
        return {"status": "error", "message": f"Issue {issue_id} not found in Sentry"}
    
    # Format the response similar to `/get-sentry-issues`
    formatted_issue = {
        "id": issue_id,
        "title": error_details.get("error_message", "No title available"),
        "permalink": f"https://sentry.io/organizations/{ORG_SLUG}/issues/{issue_id}/",
        "error_type": error_details["error_type"],
        "error_message": error_details["error_message"],
        "error_location": error_details["error_location"],
        "timestamp": error_details["timestamp"]
    }

    return {"status": "success", "issue": formatted_issue}

```python
@app.get("/sentry-debug", tags=["Sentry"])
async def trigger_error():
    try:
        division_by_zero = 1 / 0
    except ZeroDivisionError:
        return {"message": "Cannot divide by zero!"}
```
import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from crewai import Agent, Task, Crew
from pydantic import BaseModel
from typing import Dict, Any

# Load environment variables
load_dotenv()

gapi_key = os.getenv("GEMINI_API_KEY")
GITHUB_REPO_BASE = os.getenv("GITHUB_REPO_BASE")  # Root repository URL

# Initialize FastAPI
app = FastAPI()

# In-memory storage for extracted documentation and relevant docs
documentation_content = {}
relevant_docs = {}

# Pydantic model for user queries
class QueryInput(BaseModel):
    query: str
    user_context: str

# Function to fetch markdown files from GitHub
def fetch_markdown_files(repo_url: str, folder_path="") -> Dict[str, str]:
    """
    Fetches markdown files from a GitHub repository and returns structured content.
    """
    docs_content = {}
    current_repo_url = f"{repo_url}/contents/{folder_path}" if folder_path else f"{repo_url}/contents"
    
    response = requests.get(current_repo_url)

    if response.status_code == 403:
        reset_time = response.headers.get('X-RateLimit-Reset')
        raise HTTPException(status_code=429, detail=f"Rate limit reached, try again after: {reset_time}")

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch files from GitHub")

    files = response.json()
    
    for file_info in files:
        if file_info['type'] == 'dir':
            new_folder_path = os.path.join(folder_path, file_info['name']) if folder_path else file_info['name']
            docs_content.update(fetch_markdown_files(repo_url, new_folder_path))
        
        elif file_info['name'].endswith(".md"):
            file_name = file_info['name']
            file_url = file_info['download_url']
            
            file_response = requests.get(file_url)
            if file_response.status_code == 200:
                docs_content[file_name] = file_response.text
            else:
                docs_content[file_name] = f"Error downloading file: {file_response.status_code}"

    return docs_content

# Function to process and store relevant documentation
def process_documentation(docs_content: Dict[str, str]):
    """
    Process the documentation content and store relevant information.
    """
    global relevant_docs
    relevant_docs = {}
    
    for doc_name, content in docs_content.items():
        # Process the content and extract key information
        doc_sections = content.split('\n\n')  # Split by double newlines to get sections
        
        relevant_docs[doc_name] = {
            'content': content,
            'sections': doc_sections,
            'metadata': {
                'title': doc_name,
                'section_count': len(doc_sections),
                'content_length': len(content)
            }
        }

# Define Agents
crawler_agent = Agent(
    role="Documentation Crawler",
    goal="Extract and structure content from Markdown documentation files.",
    backstory="Expert in reading and processing documentation files.",
    tools=[],
    verbose=True,
    memory=True,
    llm="gemini/gemini-1.5-flash-latest"
)

analyzer_agent = Agent(
    role="Content Analyzer",
    goal="Process and organize documentation content for efficient retrieval.",
    backstory="Analyzes technical documentation and creates structured knowledge bases.",
    verbose=True,
    memory=True,
    llm="gemini/gemini-1.5-flash-latest"
)

user_assistant = Agent(
    role="Documentation Guide",
    goal="Help users understand and apply documentation effectively.",
    backstory="Helps users navigate and understand documentation step-by-step.",
    verbose=True,
    memory=True,
    llm="gemini/gemini-1.5-flash-latest"
)

# Define Tasks
crawl_task = Task(
    description="Read all markdown files from the documentation directory and extract structured information.",
    expected_output="A structured dictionary of extracted content from all markdown files.",
    agent=crawler_agent,
    tools=[],
    verbose=True
)

analyze_task = Task(
    description="Process the extracted documentation content, generate summaries, and create a searchable knowledge base.",
    expected_output="A processed knowledge base with indexed content and metadata.",
    agent=analyzer_agent,
    verbose=True
)

assist_task = Task(
    description="Answer user queries based on documentation, providing clear explanations and troubleshooting help.",
    expected_output="Detailed answers with step-by-step guidance and documentation references.",
    agent=user_assistant,
    verbose=True
)

# Embedder configuration
embedder_config = {
    "provider": "google",
    "config": {
        "api_key": gapi_key,
        "model": "models/embedding-001"
    }
}

# Create the CrewAI team
crew = Crew(
    agents=[crawler_agent, analyzer_agent, user_assistant],
    tasks=[crawl_task, analyze_task, assist_task],
    verbose=True,
    memory=True,
    embedder=embedder_config
)

# FastAPI Endpoints

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI GitHub Documentation Crawler!"}

@app.get("/fetch_docs")
def fetch_docs():
    """
    Fetch and process markdown documentation from a GitHub repository.
    """
    global documentation_content
    if not GITHUB_REPO_BASE:
        raise HTTPException(status_code=500, detail="GitHub repository base URL is missing in environment variables.")

    documentation_content = fetch_markdown_files(GITHUB_REPO_BASE)
    # Process and store relevant documentation
    process_documentation(documentation_content)
    
    return {
        "message": "Documentation fetched and processed successfully",
        "content": documentation_content,
        "relevant_docs_count": len(relevant_docs)
    }

@app.get("/docs_content")
def get_docs_content():
    """
    Retrieve the extracted documentation content.
    """
    if not documentation_content:
        raise HTTPException(status_code=404, detail="No documentation content found. Please fetch first.")

    return documentation_content

@app.post("/ask_doc_assistant")
async def ask_doc_assistant(query_input: QueryInput):
    """
    Process a user's query using CrewAI's knowledge base and relevant documentation content.
    """
    if not relevant_docs:
        raise HTTPException(status_code=404, detail="No relevant documentation found. Please fetch documentation first.")

    try:
        # Find relevant documentation sections based on the query
        query_terms = query_input.query.lower().split()
        matching_docs = {}
        
        for doc_name, doc_info in relevant_docs.items():
            content = doc_info['content'].lower()
            if any(term in content for term in query_terms):
                matching_docs[doc_name] = doc_info

        if not matching_docs:
            return {
                "response": "No relevant documentation found for your query.",
                "matching_docs": []
            }

        # Create specific tasks for handling this query with relevant docs
        analyze_query_task = Task(
            description=f"""
                Analyze the user's query and identify relevant sections in the documentation.
                User Context: {query_input.user_context}
                Query: {query_input.query}
                Available Documentation: {json.dumps(matching_docs)}
            """,
            expected_output="A detailed analysis of the query with identified relevant documentation sections.",
            agent=analyzer_agent
        )

        find_answer_task = Task(
            description=f"""
                Search through the relevant documentation and compile information to answer the query.
                Provide specific references to documentation sections where the information was found.
                User Query: {query_input.query}
                Relevant Documentation: {json.dumps(matching_docs)}
            """,
            expected_output="A comprehensive answer with references to specific documentation sections.",
            agent=user_assistant
        )

        # Create a new crew for this specific query
        query_crew = Crew(
            agents=[analyzer_agent, user_assistant],
            tasks=[analyze_query_task, find_answer_task],
            verbose=True
        )

        # Execute the tasks
        result = query_crew.kickoff()

        # Structure the response
        response = {
            "response": result,
            "query_context": {
                "original_query": query_input.query,
                "user_context": query_input.user_context
            },
            "matching_docs": list(matching_docs.keys())
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/save_docs")
def save_docs_to_file():
    """
    Save both documentation content and relevant docs to JSON files.
    """
    if not documentation_content or not relevant_docs:
        raise HTTPException(status_code=404, detail="No documentation content available to save.")

    # Save documentation content
    docs_file_path = "documentation_content.json"
    with open(docs_file_path, "w", encoding="utf-8") as json_file:
        json.dump(documentation_content, json_file, ensure_ascii=False, indent=4)

    # Save relevant docs
    relevant_docs_file_path = "relevant_docs.json"
    with open(relevant_docs_file_path, "w", encoding="utf-8") as json_file:
        json.dump(relevant_docs, json_file, ensure_ascii=False, indent=4)

    return {
        "message": "Documentation saved successfully",
        "files": {
            "documentation": docs_file_path,
            "relevant_docs": relevant_docs_file_path
        }
    }

# Run the FastAPI app with:
# uvicorn main:app --reload
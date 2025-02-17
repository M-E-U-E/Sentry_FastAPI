import os
import requests
import base64
import json
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from crewai import Agent, Task, Crew
from crewai.tools import tool
from litellm import GoogleAIStudioGeminiConfig
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()]
)

# FastAPI app
app = FastAPI()

# -------------------------
# ✅ Define the Pydantic Model Correctly
# -------------------------
class ErrorFixRequest(BaseModel):
    error_id: str
    error_title: str
    error_message: str
    error_location: Dict[str, str]  # Ensures correct type

# -------------------------
# ✅ GitHub API Utility Functions with Proper Docstrings
# -------------------------

@tool
def get_repository_info(repo_url: str) -> str:
    """Fetches details and contents of a GitHub repository."""
    try:
        parts = repo_url.strip("/").split("/")
        owner, repo = parts[-2], parts[-1]

        headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
        repo_api_url = f"https://api.github.com/repos/{owner}/{repo}"

        repo_response = requests.get(repo_api_url, headers=headers)
        repo_response.raise_for_status()

        contents_url = f"{repo_api_url}/contents"
        contents_response = requests.get(contents_url, headers=headers)
        contents_response.raise_for_status()

        return f"Repository Details:\n{json.dumps(repo_response.json(), indent=2)}\n\nRoot Contents:\n{json.dumps(contents_response.json(), indent=2)}"
    except requests.RequestException as e:
        logging.error(f"GitHub API Error: {str(e)}")
        return f"Error accessing repository: {str(e)}"

@tool
def get_file_content(repo_url: str, file_path: str) -> str:
    """Fetches the content of a file from a GitHub repository."""
    try:
        parts = repo_url.strip("/").split("/")
        owner, repo = parts[-2], parts[-1]

        headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
        file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"

        response = requests.get(file_api_url, headers=headers)
        response.raise_for_status()

        file_data = response.json()
        if file_data.get("type") != "file":
            return f"Invalid file path: {file_path}"

        return base64.b64decode(file_data["content"]).decode("utf-8")
    except requests.RequestException as e:
        logging.error(f"GitHub API Error: {str(e)}")
        return f"Error accessing file: {str(e)}"

@tool
def update_file_content(repo_url: str, file_path: str, new_content: str, commit_message: str) -> str:
    """Updates a file in a GitHub repository with new content."""
    try:
        parts = repo_url.strip("/").split("/")
        owner, repo = parts[-2], parts[-1]

        headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
        file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"

        response = requests.get(file_api_url, headers=headers)
        response.raise_for_status()

        file_data = response.json()
        file_sha = file_data["sha"]

        update_payload = {
            "message": commit_message,
            "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
            "sha": file_sha
        }

        update_response = requests.put(file_api_url, headers=headers, json=update_payload)
        update_response.raise_for_status()

        return f"File {file_path} updated successfully with commit: {commit_message}"
    except requests.RequestException as e:
        logging.error(f"GitHub API Error: {str(e)}")
        return f"Error updating file: {str(e)}"

# -------------------------
# ✅ Sentry CrewAI Fixer Class (Modular and Optimized)
# -------------------------

class SentryCrewFixer:
    """Handles automated error diagnosis and fixing for FastAPI + Sentry applications."""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_PAT")
        self.google_api_key = os.getenv("GEMINI_API_KEY")

    def create_gemini_llm(self, temperature=0.2):
        """Creates a Gemini AI model with specified temperature settings."""
        # Check if API key and other arguments are passed differently.
        config = GoogleAIStudioGeminiConfig(
            model="models/gemini-2.0-flash", 
            temperature=temperature,
            provider="google",  
            api_key=self.google_api_key  
        )

        return config

    def create_agents(self):
        """Creates CrewAI agents for error analysis and fixing."""
        return [
            Agent(
                role="Code Analyzer",
                goal="Analyze Sentry FastAPI codebase and identify issues",
                backstory="Expert in FastAPI and Sentry error diagnostics.",
                verbose=True,
                llm="gemini/gemini-2.0-flash"
            ),
            Agent(
                role="Root Cause Investigator",
                goal="Identify root causes of Sentry-reported errors",
                backstory="Specializes in tracing FastAPI errors to their source.",
                verbose=True,
                llm="gemini/gemini-2.0-flash"
            ),
            Agent(
                role="Solution Architect",
                goal="Design fixes for FastAPI and Sentry integration issues",
                backstory="Senior FastAPI architect with deep Sentry knowledge.",
                verbose=True,
                llm="gemini/gemini-2.0-flash"
            ),
            Agent(
                role="Code Implementer",
                goal="Implement fixes for Sentry-reported issues",
                backstory="FastAPI expert implementing precise fixes.",
                verbose=True,
                llm="gemini/gemini-2.0-flash"
            ),
        ]


    def fix_error(self, request_data: Dict):
        """Executes CrewAI workflow to diagnose and fix an error."""
        try:
            agents = self.create_agents()
            tasks = [
                Task(
                    description=f"Analyze error {request_data['error_id']} in {request_data['error_location']['file']}.",
                    expected_output="A detailed analysis of possible issues in the FastAPI code.",
                    agent=agents[0]
                ),
                Task(
                    description=f"Trace root cause of {request_data['error_id']}.",
                    expected_output="Identification of the exact failure point in the code.",
                    agent=agents[1]
                ),
                Task(
                    description=f"Propose a fix for {request_data['error_id']}.",
                    expected_output="A recommended code fix based on best practices.",
                    agent=agents[2]
                ),
                Task(
                    description=f"Implement and test fix for {request_data['error_id']}.",
                    expected_output="Updated code with applied fixes and successful test results.",
                    agent=agents[3]
                ),
            ]

            crew = Crew(agents=agents, tasks=tasks, verbose=True)
            return crew.kickoff()
        except Exception as e:
            logging.error(f"Error in fix_error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# ✅ FastAPI Endpoint (Fixed)
# -------------------------

@app.post("/crew-ai/fix-error")
async def fix_sentry_error(request: ErrorFixRequest):
    """API endpoint to analyze and fix Sentry-reported errors."""
    fixer = SentryCrewFixer()
    return {"status": "success", "fix_result": fixer.fix_error(request.model_dump())}

# -------------------------
# ✅ Start FastAPI
# -------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
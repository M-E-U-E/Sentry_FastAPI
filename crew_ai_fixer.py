import os
import re
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
# ✅ GitHub API Utility Functions
# -------------------------

def get_latest_branch_commit(owner, repo, branch):
    """Gets the latest commit SHA of a branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{branch}"
    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        logging.warning(f"Branch '{branch}' not found. Creating a new one.")
        return None  # Indicates branch does not exist
    response.raise_for_status()
    return response.json()["object"]["sha"]

def create_new_branch(owner, repo, branch):
    """Creates a new branch from the latest main branch commit."""
    base_branch = "main"
    latest_commit = get_latest_branch_commit(owner, repo, base_branch)

    if latest_commit is None:
        raise ValueError(f"Base branch '{base_branch}' does not exist in {repo}.")

    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
    branch_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
    new_branch_data = {
        "ref": f"refs/heads/{branch}",
        "sha": latest_commit
    }
    response = requests.post(branch_url, headers=headers, json=new_branch_data)
    response.raise_for_status()
    logging.info(f"Created new branch: {branch}")

def get_default_branch(owner, repo):
    """Fetches the default branch of the repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    repo_data = response.json()
    
    return repo_data.get("default_branch", "main")  # Fallback to "main" if not found

def push_updated_code(repo_base_url, file_path, new_content, commit_message, branch):
    """Pushes updated code to a dynamically generated branch and creates a pull request."""
    parts = repo_base_url.strip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}

    # ✅ Fetch the repository’s default branch dynamically
    base_branch = get_default_branch(owner, repo)
    logging.info(f"Default branch of {repo}: {base_branch}")

    # ✅ Ensure branch exists
    if get_latest_branch_commit(owner, repo, branch) is None:
        create_new_branch(owner, repo, branch)

    # ✅ Get file details (Ensure file exists)
    file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    response = requests.get(file_api_url, headers=headers)

    if response.status_code == 404:
        raise ValueError(f"File '{file_path}' not found in branch '{branch}'.")

    response.raise_for_status()
    file_data = response.json()
    file_sha = file_data["sha"]

    # ✅ Update file content
    update_payload = {
        "message": commit_message,
        "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
        "sha": file_sha,
        "branch": branch
    }

    update_response = requests.put(file_api_url, headers=headers, json=update_payload)
    update_response.raise_for_status()
    logging.info(f"File {file_path} updated successfully on branch {branch}.")

    # ✅ Check if a PR already exists
    pr_list_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?head={owner}:{branch}"
    pr_list_response = requests.get(pr_list_url, headers=headers)

    if pr_list_response.status_code == 200 and pr_list_response.json():
        existing_pr = pr_list_response.json()[0]["html_url"]
        logging.info(f"PR already exists: {existing_pr}")
        return f"PR already exists: {existing_pr}"

    # ✅ Create pull request with the correct base branch
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    pr_payload = {
        "title": f"Fix: {commit_message}",
        "body": "This PR contains a fix for the identified issue.",
        "head": branch,
        "base": base_branch  # ✅ Use the dynamically fetched default branch
    }

    pr_response = requests.post(pr_url, headers=headers, json=pr_payload)

    if pr_response.status_code == 422:
        logging.error(f"GitHub API PR creation failed: {pr_response.json()}")
        raise ValueError(f"GitHub PR creation failed: {pr_response.json()}")

    pr_response.raise_for_status()
    return f"Pull request created: {pr_response.json().get('html_url')}"


# -------------------------
# ✅ Sentry CrewAI Fixer Class (Stronger Agent for Issue Resolution)
# -------------------------

class SentryCrewFixer:
    """Handles automated error diagnosis and fixing for FastAPI + Sentry applications."""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_PAT")
        self.google_api_key = os.getenv("GEMINI_API_KEY")

    def create_gemini_llm(self, temperature=0.2):
        """Creates a Gemini AI model with specified temperature settings."""
        config = GoogleAIStudioGeminiConfig(
            model="models/gemini-2.0-flash", 
            temperature=temperature,
            provider="google",  
            api_key=self.google_api_key  
        )
        return config

    def create_agents(self):
        """Creates a CrewAI agent for robust error analysis and fixing."""
        return [
            Agent(
                role="Advanced Error Diagnoser and Fixer",
                goal="Deeply analyze FastAPI errors, pinpoint the root cause, and implement a full fix.",
                backstory="Senior AI-powered software engineer with a decade of experience in Python debugging, backend optimization, and error handling. Specializes in critical bug fixing, ensuring system reliability, and preventing regression errors.",
                verbose=True,
                llm="gemini/gemini-2.0-flash"
            )
        ]

    def fix_error(self, request_data: Dict):
        """Executes CrewAI workflow to diagnose and fix an error."""
        try:
            agents = self.create_agents()
            tasks = [
                Task(
                    description=(
                        f"Analyze and fix the issue in {request_data['error_location']['file']}.\n"
                        f"Error details: {request_data['error_title']} - {request_data['error_message']}.\n"
                        "Steps:\n"
                        "1. Identify the exact root cause in the given file.\n"
                        "2. Ensure the issue is fully fixed.\n"
                        "3. Return only the corrected code.\n"
                        "4. The fix should be robust, tested, and should prevent similar errors."
                    ),
                    expected_output="Only return the corrected code, ensuring the issue is resolved completely.",
                    agent=agents[0]
                )
            ]

            crew = Crew(agents=agents, tasks=tasks, verbose=True)
            crew_output = crew.kickoff()
            
            # Extract results from CrewOutput
            results = list(crew_output)  # Convert CrewOutput to a list

            if len(results) < 1:
                raise ValueError("CrewAI output is missing expected results.")

            # ✅ Ensure `fixed_code` is a string and not a tuple
            fixed_code = results[0]
            if isinstance(fixed_code, tuple):
                fixed_code = fixed_code[0]  # Extract first element if it's a tuple
            fixed_code = str(fixed_code)  # Convert to string just in case


            # ✅ Auto-generate the repository URL using error_id
            repo_owner = "M-E-U-E"
            repo_name = "Sentry_FastAPI"
            repo_base_url = f"https://github.com/{repo_owner}/{repo_name}"
            error_id = request_data.get("error_id")

            if not error_id:
                raise ValueError("Request data must include an 'error_id'.")

            file_path = request_data["error_location"].get("file")

            if not file_path:
                raise ValueError("Error location must include a 'file' key.")

            # ✅ Push the fixed code to GitHub on the dynamically generated branch
            commit_message = f"Fix: {request_data['error_title']}"
            branch_name = error_id  # Use error_id as the branch name

            pr_url = push_updated_code(repo_base_url, file_path, fixed_code, commit_message, branch_name)
            return {"status": "success", "pull_request": pr_url}

        except Exception as e:
            logging.error(f"Error in fix_error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# ✅ FastAPI Endpoint
# -------------------------

@app.post("/crew-ai/fix-error")
async def fix_sentry_error(request: ErrorFixRequest):
    """API endpoint to analyze and fix Sentry-reported errors."""
    fixer = SentryCrewFixer()
    return fixer.fix_error(request.model_dump())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

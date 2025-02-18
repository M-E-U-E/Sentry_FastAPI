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

class ErrorFixRequest(BaseModel):
    error_id: str
    error_title: str
    error_message: str
    error_location: Dict[str, str]

# GitHub API Tools
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

# Direct GitHub API function for getting file content
def fetch_file_content(repo_url: str, file_path: str) -> str:
    """Directly fetches file content without using the tool decorator."""
    try:
        parts = repo_url.strip("/").split("/")
        owner, repo = parts[-2], parts[-1]
        headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
        file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        response = requests.get(file_api_url, headers=headers)
        response.raise_for_status()
        file_data = response.json()
        if file_data.get("type") != "file":
            raise ValueError(f"Invalid file path: {file_path}")
        return base64.b64decode(file_data["content"]).decode("utf-8")
    except requests.RequestException as e:
        logging.error(f"GitHub API Error: {str(e)}")
        raise ValueError(f"Error accessing file: {str(e)}")

# [Previous GitHub API Utility Functions remain the same]
def get_latest_branch_commit(owner, repo, branch):
    """Gets the latest commit SHA of a branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{branch}"
    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 404:
        logging.warning(f"Branch '{branch}' not found. Creating a new one.")
        return None
    response.raise_for_status()
    return response.json()["object"]["sha"]

def create_new_branch(owner, repo, branch):
    """Creates a new branch from the latest commit of the default branch."""
    # First, get the default branch name
    default_branch = get_default_branch(owner, repo)
    logging.info(f"Using default branch '{default_branch}' for {repo}")
    
    # Get the latest commit from the default branch
    latest_commit = get_latest_branch_commit(owner, repo, default_branch)

    if latest_commit is None:
        # Try 'master' if default branch lookup failed or returned none
        logging.warning(f"Default branch '{default_branch}' not found. Trying 'master' branch.")
        latest_commit = get_latest_branch_commit(owner, repo, "master")
        
        if latest_commit is None:
            raise ValueError(f"Could not find a valid base branch in {repo}. Checked '{default_branch}' and 'master'.")

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
    
    return repo_data.get("default_branch", "main")

def push_updated_code(repo_base_url, file_path, new_content, commit_message, branch):
    """Pushes updated code to a dynamically generated branch and creates a pull request."""
    parts = repo_base_url.strip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}

    # Get the default branch - more robust approach
    try:
        base_branch = get_default_branch(owner, repo)
        logging.info(f"Default branch of {repo}: {base_branch}")
    except Exception as e:
        logging.warning(f"Error getting default branch: {str(e)}. Falling back to 'master'.")
        base_branch = "master"

    # Try to get the latest commit - if branch exists, if not create it
    try:
        if get_latest_branch_commit(owner, repo, branch) is None:
            create_new_branch(owner, repo, branch)
    except Exception as e:
        logging.error(f"Error creating branch: {str(e)}")
        raise

    # Get the file content from the specified branch
    file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    response = requests.get(file_api_url, headers=headers)

    # If file doesn't exist in the branch yet, get it from the base branch
    if response.status_code == 404:
        logging.info(f"File '{file_path}' not found in branch '{branch}', trying base branch '{base_branch}'")
        base_file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={base_branch}"
        base_response = requests.get(base_file_api_url, headers=headers)
        
        if base_response.status_code == 404:
            # If file doesn't exist in base branch either, we'll create a new file
            file_sha = None
        else:
            base_response.raise_for_status()
            file_data = base_response.json()
            file_sha = None  # No SHA needed for new file in the branch
    else:
        response.raise_for_status()
        file_data = response.json()
        file_sha = file_data["sha"]

    # Prepare update payload
    update_payload = {
        "message": commit_message,
        "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
        "branch": branch
    }
    
    # Include sha only if we're updating an existing file
    if file_sha:
        update_payload["sha"] = file_sha

    update_response = requests.put(file_api_url, headers=headers, json=update_payload)
    update_response.raise_for_status()
    logging.info(f"File {file_path} updated successfully on branch {branch}.")

    # Check if PR already exists
    pr_list_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?head={owner}:{branch}"
    pr_list_response = requests.get(pr_list_url, headers=headers)

    if pr_list_response.status_code == 200 and pr_list_response.json():
        existing_pr = pr_list_response.json()[0]["html_url"]
        logging.info(f"PR already exists: {existing_pr}")
        return f"PR already exists: {existing_pr}"

    # Create new PR
    pr_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    pr_payload = {
        "title": f"Fix: {commit_message}",
        "body": "This PR contains a fix for the identified issue.",
        "head": branch,
        "base": base_branch
    }

    pr_response = requests.post(pr_url, headers=headers, json=pr_payload)

    if pr_response.status_code == 422:
        logging.error(f"GitHub API PR creation failed: {pr_response.json()}")
        raise ValueError(f"GitHub PR creation failed: {pr_response.json()}")

    pr_response.raise_for_status()
    return f"Pull request created: {pr_response.json().get('html_url')}"

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
        error_fixer = Agent(
            role="Advanced Error Diagnoser and Fixer",
            goal="Deeply analyze FastAPI errors, pinpoint the root cause, and implement a full fix.",
            backstory="Senior AI-powered software engineer with a decade of experience in Python debugging, backend optimization, and error handling. Specializes in critical bug fixing, ensuring system reliability, and preventing regression errors.",
            verbose=True,
            llm="gemini/gemini-2.0-flash",
            tools=[get_repository_info, get_file_content]
        )
        return [error_fixer]

    def fix_error(self, request_data: Dict):
        """Executes CrewAI workflow to diagnose and fix an error."""
        try:
            # Generate repository URL
            repo_owner = "M-E-U-E"
            repo_name = "Sentry_FastAPI"
            repo_base_url = f"https://github.com/{repo_owner}/{repo_name}"
            error_id = request_data.get("error_id")
            file_path = request_data["error_location"].get("file")

            if not error_id:
                raise ValueError("Request data must include an 'error_id'.")
            if not file_path:
                raise ValueError("Error location must include a 'file' key.")

            # Fetch current file content using the direct function instead of the tool
            current_code = fetch_file_content(repo_base_url, file_path)

            agents = self.create_agents()
            tasks = [
                Task(
                    description=(
                        f"Analyze and fix the issue in the following code:\n\n{current_code}\n\n"
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
            
            results = list(crew_output)

            if len(results) < 1:
                raise ValueError("CrewAI output is missing expected results.")

            # Extract the fixed code from the agent's output
            fixed_code = results[0]
            if isinstance(fixed_code, tuple):
                fixed_code = fixed_code[0]
            fixed_code = str(fixed_code)

            # Check if the output contains the "Final Answer:" marker
            if "Final Answer:" in fixed_code:
                # Extract only the code part after "Final Answer:"
                code_start = fixed_code.find("Final Answer:") + len("Final Answer:")
                fixed_code = fixed_code[code_start:].strip()
            
            # Remove any markdown code block markers if present
            fixed_code = re.sub(r'^```python\s*', '', fixed_code, flags=re.MULTILINE)
            fixed_code = re.sub(r'^```\s*$', '', fixed_code, flags=re.MULTILINE)
            
            # Push the fixed code and create PR
            commit_message = f"Fix: {request_data['error_title']}"
            branch_name = f"fix-{error_id}"  # Adding 'fix-' prefix to make branch name more descriptive
            pr_url = push_updated_code(repo_base_url, file_path, fixed_code, commit_message, branch_name)
            
            return {"status": "success", "pull_request": pr_url}

        except Exception as e:
            logging.error(f"Error in fix_error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/crew-ai/fix-error")
async def fix_sentry_error(request: ErrorFixRequest):
    """API endpoint to analyze and fix Sentry-reported errors."""
    fixer = SentryCrewFixer()
    return fixer.fix_error(request.model_dump())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
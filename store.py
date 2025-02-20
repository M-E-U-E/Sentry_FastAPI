import os
import re
import requests
import base64
import json
import logging
import subprocess
import datetime  # Newly added import

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
        return (
            f"Repository Details:\n{json.dumps(repo_response.json(), indent=2)}\n\n"
            f"Root Contents:\n{json.dumps(contents_response.json(), indent=2)}"
        )
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

def get_default_branch(owner, repo):
    """Fetches the default branch of the repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    repo_data = response.json()
    return repo_data.get("default_branch", "main")

def create_new_branch(owner, repo, branch):
    """Creates a new branch from the latest commit of the default branch."""
    default_branch = get_default_branch(owner, repo)
    logging.info(f"Using default branch '{default_branch}' for {repo}")
    latest_commit = get_latest_branch_commit(owner, repo, default_branch)

    if latest_commit is None:
        logging.warning(f"Default branch '{default_branch}' not found. Trying 'master' branch.")
        latest_commit = get_latest_branch_commit(owner, repo, "master")
        if latest_commit is None:
            raise ValueError(
                f"Could not find a valid base branch in {repo}. "
                f"Checked '{default_branch}' and 'master'."
            )

    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}
    branch_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
    new_branch_data = {
        "ref": f"refs/heads/{branch}",
        "sha": latest_commit
    }
    response = requests.post(branch_url, headers=headers, json=new_branch_data)
    response.raise_for_status()
    logging.info(f"Created new branch: {branch}")

def push_updated_code(repo_base_url, file_path, new_content, commit_message, branch):
    """Pushes updated code via the GitHub API and creates a pull request."""
    parts = repo_base_url.strip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    headers = {"Authorization": f"token {os.getenv('GITHUB_PAT')}"}

    try:
        base_branch = get_default_branch(owner, repo)
        logging.info(f"Default branch of {repo}: {base_branch}")
    except Exception as e:
        logging.warning(f"Error getting default branch: {str(e)}. Falling back to 'master'.")
        base_branch = "master"

    try:
        if get_latest_branch_commit(owner, repo, branch) is None:
            create_new_branch(owner, repo, branch)
    except Exception as e:
        logging.error(f"Error creating branch: {str(e)}")
        raise

    file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"
    response = requests.get(file_api_url, headers=headers)

    if response.status_code == 404:
        logging.info(
            f"File '{file_path}' not found in branch '{branch}', "
            f"trying base branch '{base_branch}'"
        )
        base_file_api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={base_branch}"
        base_response = requests.get(base_file_api_url, headers=headers)
        
        if base_response.status_code == 404:
            file_sha = None
        else:
            base_response.raise_for_status()
            file_sha = None
    else:
        response.raise_for_status()
        file_data = response.json()
        file_sha = file_data["sha"]

    update_payload = {
        "message": commit_message,
        "content": base64.b64encode(new_content.encode("utf-8")).decode("utf-8"),
        "branch": branch
    }
    if file_sha:
        update_payload["sha"] = file_sha

    update_response = requests.put(file_api_url, headers=headers, json=update_payload)
    update_response.raise_for_status()
    logging.info(f"File {file_path} updated successfully on branch {branch}.")

    pr_list_url = f"https://api.github.com/repos/{owner}/{repo}/pulls?head={owner}:{branch}"
    pr_list_response = requests.get(pr_list_url, headers=headers)

    if pr_list_response.status_code == 200 and pr_list_response.json():
        existing_pr = pr_list_response.json()[0]["html_url"]
        logging.info(f"PR already exists: {existing_pr}")
        return f"PR already exists: {existing_pr}"

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

def push_updated_code_local(file_path, new_content, commit_message):
    """Pushes updated code using local git CLI commands."""
    try:
        # Write the new content to the file on disk
        with open(file_path, "w") as f:
            f.write(new_content)
        logging.info(f"Updated file {file_path} on disk.")

        # Check if there are changes to commit
        diff_result = subprocess.run(["git", "diff", "--exit-code", file_path],
                                     capture_output=True, text=True)
        if diff_result.returncode == 0:
            logging.info("No changes detected in file: " + file_path)
            return "No changes to commit."

        subprocess.run(["git", "add", file_path], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        logging.info("Local commit and push successful for file: " + file_path)
        return "Local commit and push successful."

    except subprocess.CalledProcessError as e:
        logging.error(f"Git command failed: {str(e)}")
        raise ValueError(f"Git command failed: {str(e)}")

class SentryCrewFixer:
    """Handles automated error diagnosis and fixing for FastAPI + Sentry applications."""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_PAT")
        self.google_api_key = os.getenv("GEMINI_API_KEY")

    def create_gemini_llm(self, temperature=0.2):
        """Creates a Gemini AI model with specified temperature settings."""
        config = GoogleAIStudioGeminiConfig(
            model="models/gemini/gemini-2.0-flash", 
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

    def save_agent_output(self, error_id: str, agent_output: str):
        """Saves the raw agent output to a local file for debugging and record keeping."""
        output_dir = "agent_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"agent_output_{error_id}_{timestamp}.txt"
        output_filepath = os.path.join(output_dir, output_filename)
        with open(output_filepath, "w") as f:
            f.write(agent_output)
        logging.info(f"Agent output saved to local file: {output_filepath}")

    def fix_error(self, request_data: Dict):
        """Executes CrewAI workflow to diagnose and fix an error."""
        try:
            # Generate repository URL and file details
            repo_owner = "M-E-U-E"
            repo_name = "Sentry_FastAPI"
            repo_base_url = f"https://github.com/{repo_owner}/{repo_name}"
            error_id = request_data.get("error_id")
            file_path = request_data["error_location"].get("file")

            if not error_id:
                raise ValueError("Request data must include an 'error_id'.")
            if not file_path:
                raise ValueError("Error location must include a 'file' key.")

            # 1. Fetch current file content
            current_code = fetch_file_content(repo_base_url, file_path)

            # 2. Run the agent
            agents = self.create_agents()
            tasks = [
                Task(
                    description=(
                        f"Analyze and fix the issue in the following code:\n\n{current_code}\n\n"
                        f"Error details: {request_data['error_title']} - {request_data['error_message']}.\n"
                        "Steps:\n"
                        "1. Identify the exact root cause in the given file.\n"
                        "2. Implement a complete fix that addresses the issue.\n"
                        "3. Return the ENTIRE fixed file with your changes integrated - not just a snippet.\n"
                        "4. The fix should be robust, tested, and should prevent similar errors.\n"
                        "5. Make sure to maintain all imports and functionality from the original code.\n"
                        "6. Output your final code inside a Python code block. (e.g. ```python ... ```)"
                    ),
                    expected_output="Return the complete fixed code file with your changes properly integrated.",
                    agent=agents[0]
                )
            ]
            crew = Crew(agents=agents, tasks=tasks, verbose=True)
            crew_output = crew.kickoff()
            results = list(crew_output)

            if len(results) < 1:
                raise ValueError("CrewAI output is missing expected results.")

            # 3. Extract the agent's raw output
            fixed_code = results
            if isinstance(fixed_code, tuple):
                fixed_code = fixed_code[0]
            fixed_code = str(fixed_code)

            # 4. Log the raw output for debugging
            logging.debug("Raw agent output:\n" + fixed_code)
            # Save the raw agent output to local storage
            self.save_agent_output(error_id, fixed_code)

            # 5. Try capturing any triple-backtick code block (bash or python)
            match = re.search(r'```(?:bash|python)?\s*(.*?)```', fixed_code, re.DOTALL)
            if match:
                final_code = match.group(1).strip()
            else:
                # 6. If that fails, see if there's a "Final Answer:" pattern
                if "Final Answer:" in fixed_code:
                    remainder = fixed_code.split("Final Answer:", 1)[1]
                    match = re.search(r'```(?:bash|python)?\s*(.*?)```', remainder, re.DOTALL)
                    if match:
                        final_code = match.group(1).strip()
                    else:
                        final_code = remainder.strip()
                else:
                    # 7. Final fallback: use the entire string
                    final_code = fixed_code.strip()

            # 8. Log the final code
            logging.debug("Final code to be pushed:\n" + final_code)

            # 9. Check if the agent gave us only "raw"
            if final_code.strip().lower() == "raw":
                logging.error("Agent returned only 'raw'; code extraction failed.")
                raise ValueError("No valid code block found in agent output.")

            # 10. Compare with current file
            if final_code.strip() == current_code.strip():
                logging.info("No changes detected in the code. Skipping commit.")
                return {"status": "no_changes", "message": "No changes needed in the file."}

            # 11. Push the changes
            commit_message = f"Fix: {request_data['error_title']}"
            branch_name = f"fix-{error_id}"
            commit_method = os.getenv("COMMIT_METHOD", "api").lower()

            if commit_method == "cli":
                pr_or_status = push_updated_code_local(file_path, final_code, commit_message)
            else:
                pr_or_status = push_updated_code(repo_base_url, file_path, final_code, commit_message, branch_name)

            return {"status": "success", "result": pr_or_status}

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

import requests
import os

def check_github_rate_limit():
    url = "https://api.github.com/rate_limit"
    
    github_token = os.getenv('GITHUB_TOKEN')
    
    if not github_token:
        print("❌ No GitHub Token Found. Please set the GITHUB_TOKEN environment variable.")
        return

    headers = {'Authorization': f'token {github_token}'}
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        rate_limit = data.get("rate", {})
        
        print("=== GitHub API Rate Limits ===")
        print(f"Total Limit: {rate_limit.get('limit', 'N/A')}")
        print(f"Remaining: {rate_limit.get('remaining', 'N/A')}")
        print(f"Reset Time (Unix Timestamp): {rate_limit.get('reset', 'N/A')}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

check_github_rate_limit()

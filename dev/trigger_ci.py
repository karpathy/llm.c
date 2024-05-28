import requests
# to get your GitHub personal access token
# login to GitHub, click on your profile picture, then select
# Settings / Developer settings / Tokens (Classic) / Generate new token
# and select the "repo" scope AND the "workflow" scope
#
# Replace these variables with your actual values below
GITHUB_PAT = 'YOUR_GITHUB_CLASSIC_PERSONAL_ACCESS_TOKEN'
GITHUB_USERNAME = 'your-username'
REPO = 'your-repository-name'
WORKFLOW_FILENAME = 'ci.yml'  # You can also use the workflow ID
BRANCH_NAME = 'branch-name-to-trigger-workflow-on'

url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{REPO}/actions/workflows/{WORKFLOW_FILENAME}/dispatches'
headers = {
    'Authorization': f'token {GITHUB_PAT}',
    'Accept': 'application/vnd.github.v3+json'
}
data = {
    'ref': BRANCH_NAME
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 204:
    print('Workflow dispatched successfully!')
else:
    print(f'Failed to dispatch workflow: {response.status_code}')
    print(response.json())

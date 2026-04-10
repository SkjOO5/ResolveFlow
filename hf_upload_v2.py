"""
Verify HF repo and upload with proper setup
"""

from huggingface_hub import HfApi, create_repo
import os

token = os.environ.get("HF_TOKEN", "")
api = HfApi()

print("=" * 70)
print("VERIFYING HF REPO AND UPLOADING")
print("=" * 70)

try:
    # Get authenticated user info
    user_info = api.whoami(token=token)
    print(f"\n✓ Authenticated as: {user_info['name']}")
except Exception as e:
    print(f"\n✗ Authentication failed: {e}")
    exit(1)

repo_id = "Electron005/ResolveFlow"
repo_type = "space"

print(f"\nTarget repo: {repo_id} (type: {repo_type})")

try:
    # Check if repo exists
    repo_info = api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
    print(f"✓ Repo exists: {repo_info.repo_id}")
except Exception as e:
    print(f"✗ Repo check failed: {e}")
    print("\nTrying to create repo...")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            private=False
        )
        print("✓ Repo created")
    except Exception as e2:
        print(f"✗ Create failed: {e2}")
        exit(1)

# Now upload files
files_to_upload = [
    ("envs/scoring.py", "envs/scoring.py"),
    ("envs/graders.py", "envs/graders.py"),
    ("app.py", "app.py"),
    ("tests/test_env.py", "tests/test_env.py")
]

print(f"\nUploading {len(files_to_upload)} files...\n")

for local_path, remote_path in files_to_upload:
    if not os.path.exists(local_path):
        print(f"✗ File not found: {local_path}")
        continue
        
    try:
        print(f"  {remote_path}...", end=" ")
        with open(local_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message="Fix OpenEnv validation: strict scoring and task endpoint"
        )
        print("✓")
    except Exception as e:
        print(f"✗ {e}")

print("\n" + "=" * 70)
print("UPLOAD COMPLETE")
print("=" * 70)
print(f"\nSpace: https://huggingface.co/spaces/{repo_id}")
print("Status: Files uploaded, Docker rebuild in progress...")
print("\nExpected time: 2-3 minutes")

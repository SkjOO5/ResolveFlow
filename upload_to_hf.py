"""
HF Spaces direct upload with token
"""

from huggingface_hub import HfApi
import os

# Authenticate with token
api = HfApi()
token = os.environ.get("HF_TOKEN", "")

print("=" * 70)
print("UPLOADING TO HUGGING FACE SPACES")
print("=" * 70)

repo_id = "Electron005/ResolveFlow"
repo_type = "space"

files_to_upload = [
    ("envs/scoring.py", "envs/scoring.py"),
    ("envs/graders.py", "envs/graders.py"),
    ("app.py", "app.py"),
    ("tests/test_env.py", "tests/test_env.py")
]

print(f"\nRepository: {repo_id}")
print(f"Files to upload: {len(files_to_upload)}\n")

for local_path, remote_path in files_to_upload:
    try:
        if os.path.exists(local_path):
            print(f"Uploading {local_path}...", end=" ")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token,
                commit_message="Fix OpenEnv validation: strict scoring and task endpoint"
            )
            print("✓")
        else:
            print(f"✗ File not found: {local_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)

print("\n" + "=" * 70)
print("✓ SUCCESS! All files uploaded to HF Spaces")
print("=" * 70)
print(f"\nSpace URL: https://huggingface.co/spaces/Electron005/ResolveFlow")
print("\n⏳ Waiting for Docker rebuild (2-3 minutes)...")
print("Then you can resubmit to OpenEnv!")

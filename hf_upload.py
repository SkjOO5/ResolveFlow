"""
HF Spaces upload script - authenticates and uploads fixed files
"""

from huggingface_hub import HfApi, login
import os

# Step 1: Authenticate with HF token
print("=" * 70)
print("HUGGING FACE AUTHENTICATION")
print("=" * 70)
print("\nGo to https://huggingface.co/settings/tokens")
print("Copy your HF token and paste it below:")
print()

try:
    login()
    print("\n✓ Successfully authenticated with Hugging Face!")
except Exception as e:
    print(f"\n✗ Authentication failed: {e}")
    exit(1)

# Step 2: Upload files
print("\n" + "=" * 70)
print("UPLOADING FILES TO HF SPACES")
print("=" * 70)

api = HfApi()
repo_id = "Electron005/ResolveFlow"
repo_type = "space"

files_to_upload = [
    "envs/scoring.py",
    "envs/graders.py",
    "app.py",
    "tests/test_env.py"
]

print(f"\nRepository: {repo_id}")
print(f"Type: {repo_type}")
print(f"\nFiles to upload:")

for file_path in files_to_upload:
    print(f"  - {file_path}")

print("\nUploading...")

for file_path in files_to_upload:
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Update: {file_path} - OpenEnv validation fixes"
        )
        print(f"  ✓ {file_path}")
    except Exception as e:
        print(f"  ✗ {file_path}: {e}")
        exit(1)

print("\n" + "=" * 70)
print("✓ SUCCESS! All files uploaded to HF Spaces")
print("=" * 70)
print(f"\nYour space is being rebuilt:")
print(f"https://huggingface.co/spaces/Electron005/ResolveFlow")
print("\nWait 2-3 minutes for the Docker build to complete,")
print("then you can resubmit to OpenEnv with the validation fixes!")

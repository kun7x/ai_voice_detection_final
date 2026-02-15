"""
One-click HuggingFace Space setup and deployment.

Usage:
    python setup_space.py --space_id YOUR_USERNAME/YOUR_SPACE_NAME
    python setup_space.py --space_id kun7x/detection-v3 --token hf_xxx
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Deploy AI Voice Detection to HuggingFace Spaces")
    parser.add_argument("--space_id", type=str, required=True,
                        help="HuggingFace Space ID (e.g., 'username/space-name')")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import HfApi, login

    # Login
    token = args.token or os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print(f"Logged in with provided token.")
    else:
        print("No token provided. Using cached credentials.")
        print("If not logged in, run: huggingface-cli login")

    api = HfApi()

    # Verify identity
    try:
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception:
        print("ERROR: Not authenticated. Provide --token or run: huggingface-cli login")
        sys.exit(1)

    # Create space
    try:
        api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )
        print(f"Space ready: {args.space_id}")
    except Exception as e:
        print(f"Error creating space: {e}")
        sys.exit(1)

    # Upload files (everything in this directory except this script and the guide)
    deploy_dir = os.path.dirname(os.path.abspath(__file__))
    exclude = {"setup_space.py", "DEPLOYMENT_GUIDE.md", "__pycache__", ".env"}

    print(f"\nUploading files from {deploy_dir}:")
    files_to_upload = []
    for fname in os.listdir(deploy_dir):
        if fname in exclude or fname.startswith("."):
            continue
        fpath = os.path.join(deploy_dir, fname)
        if os.path.isfile(fpath):
            files_to_upload.append(fname)
            print(f"  + {fname}")

    api.upload_folder(
        folder_path=deploy_dir,
        repo_id=args.space_id,
        repo_type="space",
        allow_patterns=files_to_upload,
    )

    space_url = f"https://huggingface.co/spaces/{args.space_id}"
    print(f"\nDone! Space deployed at: {space_url}")
    print(f"API endpoint: https://{args.space_id.replace('/', '-')}.hf.space/api/voice-detection")
    print("\nThe space will take a few minutes to build and load the model.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Upload PoseMamba-space to Hugging Face Spaces."""
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

SPACE = "nankingjings/PoseMamba-Demo"
DIR = Path(__file__).resolve().parents[2] / "PoseMamba-space"

if __name__ == "__main__":
    from huggingface_hub import login
    login(add_to_git_credential=False)
    api = HfApi()
    create_repo(SPACE, repo_type="space", space_sdk="gradio", exist_ok=True)
    upload_folder(
        folder_path=str(DIR),
        repo_id=SPACE,
        repo_type="space",
        commit_message="Add PoseMamba demo gallery and Colab links",
    )
    print(f"https://huggingface.co/spaces/{SPACE}")

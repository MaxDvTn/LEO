import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_model(repo_id, folder_path, is_private=True):
    print(f"🚀 Preparing to upload to HF Repository: {repo_id}")
    api = HfApi()
    
    # Create the repository if it doesn't already exist
    try:
        api.create_repo(repo_id=repo_id, private=is_private, exist_ok=True, repo_type="model")
        print(f"✅ Repository {repo_id} created or already exists.")
    except Exception as e:
        print(f"⚠️ Error creating repository {repo_id}: {e}")
        return False
        
    print(f"📦 Uploading contents from {folder_path} ...")
    try:
        url = api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial push of LEO Fine-tuned PEFT adapters and tokenizer for Roverplastik"
        )
        print(f"🎉 Successfully uploaded! Model is available at: {url}")
        return True
    except Exception as e:
        print(f"❌ Error during upload to {repo_id}: {e}")
        return False

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    preferred_release_path = PROJECT_ROOT / "checkpoints" / "leo_hf_release"
    legacy_release_path = PROJECT_ROOT / "checkpoints_facebook_seamless-m4t-v2-large" / "leo_hf_release"
    local_release_path = preferred_release_path if preferred_release_path.exists() else legacy_release_path
    
    if not local_release_path.exists():
        print(f"❌ Could not find model release directory: {local_release_path}")
        sys.exit(1)
        
    repo_1 = "maxbsdv/LEO-SeamlessM4T-v2-Large-Roverplastik"
    repo_2 = "LiceoDaVinciTN/LEO-SeamlessM4T-v2-Large-Roverplastik"
    
    success_1 = upload_model(repo_1, str(local_release_path), is_private=False)
    success_2 = upload_model(repo_2, str(local_release_path), is_private=False)
    
    if success_1 and success_2:
        print("\n✨ All uploads finished successfully!")
    else:
        print("\n⚠️ Some uploads failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

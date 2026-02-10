import os
from huggingface_hub import HfApi, create_repo, add_space_variable

api = HfApi()
username = "maxbsdv" 
model_repo_id = f"{username}/leo-nllb-1.3b"
space_repo_id = f"{username}/leo-translation-hub"

print(f"🚀 Starting Deployment for user: {username}")

# --- 1. Model Deployment ---
print(f"\n📦 Check/Create Model Repo: {model_repo_id}")
try:
    create_repo(model_repo_id, repo_type="model", exist_ok=True)
    print("   ✅ Repo ready.")
except Exception as e:
    print(f"   ❌ Error creating repo: {e}")
    exit(1)

print("📤 Uploading model files...")
try:
    api.upload_folder(
        folder_path="checkpoints/leo_hf_release",
        repo_id=model_repo_id,
        repo_type="model",
        commit_message="Upload LEO fine-tuned adapters"
    )
    print("   ✅ Model upload complete!")
except Exception as e:
    print(f"   ❌ Error uploading model: {e}")
    exit(1)


# --- 2. Space Deployment ---
print(f"\n🌌 Check/Create Space Repo: {space_repo_id}")
try:
    create_repo(space_repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    print("   ✅ Space ready.")
except Exception as e:
    print(f"   ❌ Error creating space: {e}")
    exit(1)

print("📤 Uploading App files...")
try:
    # Upload README
    api.upload_file(
        path_or_fileobj="src/ui/README.md",
        path_in_repo="README.md",
        repo_id=space_repo_id,
        repo_type="space",
        commit_message="Update Space configuration"
    )
    # Upload App Script
    api.upload_file(
        path_or_fileobj="src/ui/hf_spaces_app.py",
        path_in_repo="hf_spaces_app.py",
        repo_id=space_repo_id,
        repo_type="space",
        commit_message="Update App logic"
    )
    # Upload requirements
    api.upload_file(
        path_or_fileobj="src/ui/requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=space_repo_id,
        repo_type="space",
        commit_message="Add pinned requirements"
    )
    print("   ✅ App files uploaded!")
except Exception as e:
    print(f"   ❌ Error uploading app files: {e}")

# --- 3. Configuration ---
print(f"\n⚙️ Configuring Space Variable: ADAPTER_PATH = {model_repo_id}")
try:
    # We use a variable (visible) because the model ID is public info
    add_space_variable(space_repo_id, "ADAPTER_PATH", model_repo_id)
    print("   ✅ Configuration set.")
except Exception as e:
    print(f"   ⚠️ Could not set variable automatically (might need manual set): {e}")

print("\n✨ Deployment Summary:")
print(f"   - Model: https://huggingface.co/{model_repo_id}")
print(f"   - Space: https://huggingface.co/spaces/{space_repo_id}")

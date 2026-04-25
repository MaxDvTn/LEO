import argparse
import sys
from pathlib import Path

import requests
from huggingface_hub import HfApi, add_space_variable, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

DEFAULT_MODEL_REPO = "maxbsdv/LEO-SeamlessM4T-v2-Large-Roverplastik"
DEFAULT_ORG_MODEL_REPO = "LiceoDaVinciTN/LEO-SeamlessM4T-v2-Large-Roverplastik"
DEFAULT_SPACE_REPO = "maxbsdv/leo-translation-hub"


def release_path():
    preferred = PROJECT_ROOT / "checkpoints" / "leo_hf_release"
    legacy = PROJECT_ROOT / "checkpoints_facebook_seamless-m4t-v2-large" / "leo_hf_release"
    path = preferred if preferred.exists() else legacy
    if not path.exists():
        raise FileNotFoundError(f"Could not find model release directory: {path}")
    return path


def export_model(_args):
    from scripts.hf.export_to_hf import main as export_latest_model

    export_latest_model()


def upload_model(args):
    api = HfApi()
    path = release_path()
    repos = args.repo or [DEFAULT_MODEL_REPO, DEFAULT_ORG_MODEL_REPO]

    for repo_id in repos:
        create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload LEO Seamless-M4T LoRA adapters",
        )
        print(f"Uploaded model files to https://huggingface.co/{repo_id}")


def deploy_space(args):
    api = HfApi()
    create_repo(args.space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)

    files = [
        ("src/ui/README.md", "README.md"),
        ("src/ui/hf_spaces_app.py", "hf_spaces_app.py"),
        ("src/ui/requirements.txt", "requirements.txt"),
    ]
    for local_path, remote_path in files:
        api.upload_file(
            path_or_fileobj=str(PROJECT_ROOT / local_path),
            path_in_repo=remote_path,
            repo_id=args.space_repo,
            repo_type="space",
            commit_message="Deploy LEO Translation Hub",
        )

    add_space_variable(args.space_repo, "ADAPTER_PATH", args.model_repo)
    print(f"Set ADAPTER_PATH={args.model_repo} on {args.space_repo}")
    if args.restart:
        api.restart_space(args.space_repo)
        print(f"Restarted Space {args.space_repo}")


def restart_space(args):
    HfApi().restart_space(args.space_repo)
    print(f"Restarted Space {args.space_repo}")


def smoke_test(args):
    base_url = args.url.rstrip("/")
    payload = {"data": [args.text, args.src, args.tgt]}
    response = requests.post(f"{base_url}/gradio_api/call/translate", json=payload, timeout=30)
    response.raise_for_status()
    event_id = response.json()["event_id"]
    result = requests.get(f"{base_url}/gradio_api/call/translate/{event_id}", timeout=180)
    result.raise_for_status()
    print(result.text)


def build_parser():
    parser = argparse.ArgumentParser(description="LEO Hugging Face operations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export latest checkpoint to HF adapter format")
    export_parser.set_defaults(func=export_model)

    upload_parser = subparsers.add_parser("upload-model", help="Upload adapter release to model repo(s)")
    upload_parser.add_argument("--repo", action="append", help="Model repo id. Can be passed multiple times.")
    upload_parser.add_argument("--private", action="store_true", help="Create model repo(s) as private")
    upload_parser.set_defaults(func=upload_model)

    deploy_parser = subparsers.add_parser("deploy-space", help="Upload Space files and configure ADAPTER_PATH")
    deploy_parser.add_argument("--space-repo", default=DEFAULT_SPACE_REPO)
    deploy_parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    deploy_parser.add_argument("--restart", action="store_true")
    deploy_parser.set_defaults(func=deploy_space)

    restart_parser = subparsers.add_parser("restart-space", help="Restart a Space")
    restart_parser.add_argument("--space-repo", default=DEFAULT_SPACE_REPO)
    restart_parser.set_defaults(func=restart_space)

    smoke_parser = subparsers.add_parser("smoke-test", help="Call the Gradio translation API")
    smoke_parser.add_argument("--url", default="https://maxbsdv-leo-translation-hub.hf.space")
    smoke_parser.add_argument("--text", default="This window profile ensures excellent air tightness.")
    smoke_parser.add_argument("--src", default="English")
    smoke_parser.add_argument("--tgt", default="Italian")
    smoke_parser.set_defaults(func=smoke_test)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

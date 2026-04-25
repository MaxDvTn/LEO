import argparse
import os
import subprocess


def tmux_has_session(session):
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def start_server(args):
    if tmux_has_session(args.session):
        print(f"Session '{args.session}' already exists. Attach with: tmux attach -t {args.session}")
        return

    subprocess.run(["tmux", "new-session", "-d", "-s", args.session], check=True)

    streamlit_cmd = (
        "source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {args.conda_env} && "
        f"streamlit run {args.app} --server.port {args.port}"
    )
    ngrok_cmd = f"ngrok http --domain={args.domain} {args.port}"

    subprocess.run(["tmux", "send-keys", "-t", f"{args.session}:0.0", streamlit_cmd, "C-m"], check=True)
    subprocess.run(["tmux", "split-window", "-h", "-t", f"{args.session}:0"], check=True)
    subprocess.run(["tmux", "send-keys", "-t", f"{args.session}:0.1", ngrok_cmd, "C-m"], check=True)

    print(f"Server started in tmux session '{args.session}'.")
    print(f"Attach with: tmux attach -t {args.session}")


def stop_server(args):
    if not tmux_has_session(args.session):
        print(f"Session '{args.session}' is not running.")
        return
    subprocess.run(["tmux", "kill-session", "-t", args.session], check=True)
    print(f"Stopped tmux session '{args.session}'.")


def status_server(args):
    if tmux_has_session(args.session):
        print(f"Session '{args.session}' is running.")
    else:
        print(f"Session '{args.session}' is not running.")


def attach_server(args):
    os.execvp("tmux", ["tmux", "attach", "-t", args.session])


def add_common_args(parser):
    parser.add_argument("--session", default="server")


def build_parser():
    parser = argparse.ArgumentParser(description="Manage local LEO Streamlit + ngrok server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start Streamlit and ngrok in tmux")
    add_common_args(start_parser)
    start_parser.add_argument("--conda-env", default="LEO")
    start_parser.add_argument("--app", default="src/ui/app.py")
    start_parser.add_argument("--port", default="8501")
    start_parser.add_argument("--domain", default="concretely-dendroid-florinda.ngrok-free.dev")
    start_parser.set_defaults(func=start_server)

    stop_parser = subparsers.add_parser("stop", help="Stop the tmux server session")
    add_common_args(stop_parser)
    stop_parser.set_defaults(func=stop_server)

    status_parser = subparsers.add_parser("status", help="Show server session status")
    add_common_args(status_parser)
    status_parser.set_defaults(func=status_server)

    attach_parser = subparsers.add_parser("attach", help="Attach to the tmux server session")
    add_common_args(attach_parser)
    attach_parser.set_defaults(func=attach_server)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

import time
import subprocess
import os
import sys

def get_running_gen_pid():
    try:
        # Check for python process running run_gen.py
        # Exclude this script itself if it matched somehow (unlikely)
        res = subprocess.check_output(["pgrep", "-f", "scripts/run_gen.py"])
        pids = [int(p) for p in res.decode().strip().split('\n') if p]
        return pids
    except subprocess.CalledProcessError:
        return []

def main():
    print("⏳ Orchestrator: Waiting for data generation to complete...")
    
    while True:
        pids = get_running_gen_pid()
        if not pids:
            print("✅ Data generation finished.")
            break
        print(f"   ... Generation still running (PIDs: {pids}). Waiting 60s.")
        time.sleep(60)

    print("\n🚀 Starting Training Pipeline (20 Epochs)...")
    
    # 1. Update Test Set (just in case)
    print("   🧪 refreshing test set distribution...")
    # executing via python command to ensure environment consistency
    subprocess.run([sys.executable, "scripts/create_test_set.py"], check=False)

    # 2. Start Training
    train_cmd = [sys.executable, "scripts/run_training.py"]
    print(f"   🏋️ Executing: {' '.join(train_cmd)}")
    
    log_file = open("training_final.log", "w")
    try:
        subprocess.run(train_cmd, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        print("✅ Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}. Check training_final.log")
    finally:
        log_file.close()

if __name__ == "__main__":
    main()

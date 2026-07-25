import subprocess
import sys
import os
import time

BASE_DIR = "/Users/shahidabatool/Desktop/MRP/Task_1_Classification"
CODE_DIR = os.path.join(BASE_DIR, "Code")
PYTHON_EXEC = sys.executable

scripts = [
    # 1. Preprocessing and cleaning phase
    "preprocessing/preprocess_cough.py",
    "preprocessing/preprocess_gunshot.py",
    "preprocessing/preprocess_impact.py",
    "preprocessing/preprocess_passive.py",
    "preprocessing/preprocess_transfer.py",
    
    # 2. Augmentation and splitting phase
    "preprocessing/augment_cough.py",
    "preprocessing/augment_gunshot.py",
    "preprocessing/augment_impact.py",
    "preprocessing/augment_passive.py",
    "preprocessing/augment_transfer.py"
]

def run_script(script_name):
    script_path = os.path.join(CODE_DIR, script_name)
    print(f"\n==================================================")
    print(f"Running: {script_name}")
    print(f"==================================================")
    
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    t0 = time.time()
    res = subprocess.run([PYTHON_EXEC, script_path], cwd=CODE_DIR, env=env)
    t1 = time.time()
    
    if res.returncode != 0:
        print(f"[!] Error: {script_name} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    else:
        print(f"[+] Success: {script_name} completed in {t1 - t0:.2f} seconds.")

def main():
    t_start = time.time()
    print("Starting master BPA classification data pipeline run...")
    
    for script in scripts:
        run_script(script)
        
    print(f"\n==================================================")
    print(f"BPA PIPELINE COMPLETED IN {time.time() - t_start:.2f} seconds!")
    print(f"==================================================")

if __name__ == "__main__":
    main()

"""
=============================================================================
  JET IMPINGEMENT GNN PIPELINE — MASTER ORCHESTRATOR
  
  Executes the three main stages of the SciML pipeline in sequence:
  1. Feature Engineering (Data Split + Scaling)
  2. Dimensionality Reduction (Graph template construction)
  3. Training (Latent GNN execution)
=============================================================================
"""

import subprocess
import sys
import time
import os

SCRIPTS = [
    'FeatureEngineering.py',
    'dimen_red.py',
    'train.py'
]

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"  STARTING PHASE: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Use sys.executable to ensure we use the same environment
        process = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n {script_name} COMPLETED SUCCESSFULLY")
        print(f" Time elapsed: {elapsed/60:.2f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {script_name} failed with exit code {e.returncode}")
        print(f"  Pipeline halted.")
        return False
    except Exception as e:
        print(f"\nUNEXPECTED ERROR executing {script_name}: {e}")
        return False

if __name__ == "__main__":
    total_start = time.time()
    
    for script in SCRIPTS:
        if not os.path.exists(script):
            print(f"FATAL: Script {script} not found in current directory.")
            sys.exit(1)
            
        success = run_script(script)
        if not success:
            sys.exit(1)
            
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"FULL PIPELINE EXECUTION COMPLETE")
    print(f"Total time: {total_elapsed/60:.2f} minutes")
    print(f"{'='*60}\n")

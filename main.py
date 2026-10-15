
import subprocess
import sys

scripts = [
    "preprocessing.py",
    "eda.py",
    "clustering.py",
    "cluster_analysis.py",
    "recommendation.py"
]

for script in scripts:
    print(f"\nRunning {script}...")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"Error while running {script}.")
        break
else:
    print("\nAll project scripts completed successfully!")

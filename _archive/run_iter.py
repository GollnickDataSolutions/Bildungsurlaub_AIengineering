"""Wrapper to run coffee_shop_train.py from the correct directory with iteration tracking."""
import os
import sys
import json
import subprocess

ITERATIONS_FILE = os.path.join(os.path.dirname(__file__), "iteration_results.json")


def run_iteration(iteration_num, script_path):
    """Run the training script and capture R2 score and final losses."""
    os.chdir(os.path.dirname(script_path))
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, cwd=os.path.dirname(script_path),
    )
    return result.stdout, result.stderr


def parse_output(output):
    """Parse R2 score from stdout."""
    r2 = None
    for line in output.split("\n"):
        if "R2-Score" in line:
            r2 = float(line.split(":")[1].strip())
    return r2


if __name__ == "__main__":
    script = os.path.join(os.path.dirname(__file__), "coffee_shop_train.py")
    stdout, stderr = run_iteration(0, script)

    results = {"baseline": {}}
    r2 = parse_output(stdout)
    results["baseline"]["r2"] = r2

    print("=== STDOUT ===")
    print(stdout)
    if stderr:
        print("=== STDERR ===")
        print(stderr)
    print("\n=== PARSED RESULTS ===")
    print(f"R2 Score: {r2}")

    with open(ITERATIONS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {ITERATIONS_FILE}")

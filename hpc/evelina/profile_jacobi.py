import numpy as np
import builtins
import line_profiler
from simulate import jacobi, load_data


def run_profile():
    LOAD_DIR = 'hpc/evelina/data'

    building_ids = ["10000", "10009", "10014", "10019", "10029"]

    # Initialize the Line Profiler
    lp = line_profiler.LineProfiler()
    lp.add_function(jacobi)
    
    print(f"Profiling Jacobi across {len(building_ids)} dwellings...")

    for i in building_ids:
        try:
            # Use your existing load_data function
            u, mask = load_data(LOAD_DIR, i)

            # Wrap the call in the profiler
            # We run fewer iterations (e.g., 500) just to get the percentages quickly
            lp.runcall(jacobi, u, mask, max_iter=500, atol=1e-4)
            print(f"  Finished processing {i}")

        except FileNotFoundError:
            print(f"  Skipping {i}: files not found in {LOAD_DIR}")

    # 4. Show the aggregated results
    print("\n" + "="*30)
    print("AGGREGATED LINE-BY-LINE STATS")
    print("="*30)
    lp.print_stats()

if __name__ == "__main__":
    run_profile()

# to test the profiler, run:
# uv run python hpc/evelina/profile_jacobi.py or uv run kernprof -l -v hpc/evelina/profile_jacobi.py
# To visualise performance, run:
# uv run snakeviz jacobi_performance.prof
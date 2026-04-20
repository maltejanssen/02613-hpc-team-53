import numpy as np
import builtins
import line_profiler
from simulate import load_data

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    # Create a copy of the interior
    interior = u[1:-1, 1:-1]

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        # Check convergence every 500 iterations to save time
        if i % 500 == 0:
            # We check the max change only within the masked interior
            delta = np.abs(interior[interior_mask] - u_new[interior_mask]).max()
            if delta < atol:
                break

        # Apply u_new where mask is True, keep old where False
        interior[:] = np.where(interior_mask, u_new, interior)

    return u


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

    print("\n" + "="*30)
    print("AGGREGATED LINE-BY-LINE STATS")
    print("="*30)
    lp.print_stats()

if __name__ == "__main__":
    run_profile()

# to test the profiler, run:
# uv run python hpc/evelina/profile_jacobi.py or uv run kernprof -l -v hpc/evelina/profile_jacobi.py
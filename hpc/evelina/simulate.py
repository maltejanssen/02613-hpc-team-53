from os.path import join
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import time
import builtins
from numba import njit


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def solve_building(args):
    u0, mask, max_iter, atol = args
    # Check which process is doing the work
    pid = os.getpid()
    print(f"Process {pid} is starting a floorplan...")
    
    result = jacobi(u0, mask, max_iter, atol)
    return result

@njit
def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = u.copy()
    rows, cols = interior_mask.shape

    for i in range(max_iter):

        if i % 500 == 0:
            max_delta = 0.0
            for r in range(rows):
                for c in range(cols):
                    if interior_mask[r, c]:
                        # Standard Jacobi: (Left + Right + Up + Down) / 4
                        new_val = 0.25 * (u[r+1, c] + u[r+1, c+2] + u[r, c+1] + u[r+2, c+1])
                        diff = abs(new_val - u[r+1, c+1])
                        if diff > max_delta:
                            max_delta = diff
            
            if max_delta < atol:
                break
        for r in range(rows):
            for c in range(cols):
                if interior_mask[r, c]:
                    u[r+1, c+1] = 0.25 * (u[r+1, c] + u[r+1, c+2] + u[r, c+1] + u[r+2, c+1])

    return u

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

if __name__ == '__main__':
    # Load data
    LOAD_DIR = os.environ.get('LOAD_DIR', '/dtu/projects/02613_2025/data/modified_swiss_dwellings/')
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]
    print("Building IDs:", building_ids)

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan (or benchmark pool sizes with --bench)
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    if '--bench' in sys.argv:
        worker_tests = [1, 2, 4, 8]
        print(f"Running benchmark on N={N} buildings...")
    else:
        # If benchmarking not passed use default 4 workers
        worker_tests = [4]

    results = []
    execution_times = []

    for p in worker_tests:
        print(f"Starting execution with pool size: {p}")
        t0 = time.time()
        
        c_size = max(1, len(building_ids) // p) 

        with multiprocessing.Pool(p) as pool:
            tasks = [(all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL) for i in range(N)]
            
            results = pool.map(solve_building, tasks, chunksize=c_size)
            
        elapsed = time.time() - t0
        execution_times.append(elapsed)
        print(f"Pool {p} took: {elapsed:.3f}s")

    # Visualize the speedup
    execution_times = np.array(execution_times)
    # Speedup S = T_1 / T_p
    speedup = execution_times[0] / execution_times
    plt.figure(figsize=(8, 5))
    plt.plot(worker_tests, speedup, 'o-', label='Actual Speedup', color='teal')
    plt.plot(worker_tests, worker_tests, '--', label='Ideal (Linear) Speedup', color='gray')
    plt.xlabel('Number of Workers (p)')
    plt.ylabel('Speedup (S)')
    plt.title(f'Scalability Analysis: Jacobi Solver (N={N})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    # Save the plot in the current directory or a specific folder
    plot_path = "speedup_plot.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")

    # Print Summary Statistics
    print("\n--- Summary Statistics ---")
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header

    for i, bid in enumerate(building_ids):
        # Use the solved result from the last pool run
        stats = summary_stats(results[i], all_interior_mask[i])
        print(f"{bid},", ", ".join(f"{stats[k]:.2f}" for k in stat_keys))

    sys.exit(0)

from os.path import join
import sys
import time
import os
import cupy as cp
import numpy as np


def load_data_to_gpu(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi_cupy(u, interior_mask, max_iter, atol=1e-6, check_interval=500):
    u = cp.copy(u)
    mask_float = interior_mask.astype(cp.float64)
    u_int = u[1:-1, 1:-1]

    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        
        if i % check_interval == 0:
            delta = cp.abs(u_int[interior_mask] - u_new[interior_mask]).max()
            if delta < atol:
                u_int[interior_mask] = u_new[interior_mask]
                break
        
        u_int[:] = cp.where(interior_mask, u_new, u_int)

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = cp.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = cp.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Time
    start_time = time.time()

    # Load floor plans
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data_to_gpu(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    CHECK_ITER = 500

    all_u = cp.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_cupy(u0, interior_mask, MAX_ITER, ABS_TOL, CHECK_ITER)
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    print(f"Total time on {N} buildings: {time.time() - start_time:.2f} seconds")

    csv_path = "benchmark_results.csv"
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, "a") as f:
        # Add header only if the file is new
        if not file_exists:
            f.write("script_name,num_buildings,total_time_seconds\n")
        
        # Write the data
        f.write(f"{sys.argv[0]},{N},{time.time() - start_time}\n")

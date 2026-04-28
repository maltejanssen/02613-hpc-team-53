from os.path import join
import sys
import math
import numpy as np
from multiprocessing import Pool

LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter=20_000, atol=1e-6):
    u = np.copy(u)

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] +
            u[1:-1, 2:] +
            u[:-2, 1:-1] +
            u[2:, 1:-1]
        )
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break

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


def process_building(bid, load_dir):
    u0, interior_mask = load_data(load_dir, bid)
    u = jacobi(u0, interior_mask)
    stats = summary_stats(u, interior_mask)
    return bid, stats


def process_chunk(args):
    bids, load_dir = args
    out = []
    for bid in bids:
        out.append(process_building(bid, load_dir))
    return out


def split_static(seq, workers):
    n = len(seq)
    base = n // workers
    rem = n % workers

    chunks = []
    start = 0
    for i in range(workers):
        size = base + (1 if i < rem else 0)
        end = start + size
        chunks.append(seq[start:end])
        start = end

    return chunks


if __name__ == '__main__':
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    workers = int(sys.argv[2]) if len(sys.argv) >= 3 else 2

    building_ids = building_ids[:N]

    chunks = split_static(building_ids, workers)
    
    with Pool(processes=len(chunks)) as pool:
        chunk_results = pool.map(process_chunk, [(chunk, LOAD_DIR) for chunk in chunks])

    results = [item for chunk in chunk_results for item in chunk]

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))

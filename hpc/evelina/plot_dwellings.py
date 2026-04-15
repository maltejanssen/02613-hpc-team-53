import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Setup paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
building_id = "10000"

interior_path = os.path.join(DATA_DIR, f"{building_id}_interior.npy")
domain_path = os.path.join(DATA_DIR, f"{building_id}_domain.npy")

# 2. Load the arrays
interior_array = np.load(interior_path)
domain_array = np.load(domain_path)

# 3. Create the side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Interior
im1 = axes[0].imshow(interior_array, cmap='hot')
axes[0].set_title(f"Building {building_id}: Interior")
fig.colorbar(im1, ax=axes[0])

# Plot Domain
im2 = axes[1].imshow(domain_array, cmap='hot')
axes[1].set_title(f"Building {building_id}: Domain")
fig.colorbar(im2, ax=axes[1])

# 4. Final layout adjustments
plt.tight_layout()
plt.show()
import numpy as np
import os

BASE_DIR = os.environ.get("BASE_DIR")
d = np.load(f"{BASE_DIR}code/openscene/demo/tmp/chair.npy")

print(d.shape, np.linalg.norm(d))

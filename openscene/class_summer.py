import numpy as np
import pandas as pd
from pprint import pprint
import os
BASE_DIR = os.environ.get("BASE_DIR")
df = pd.read_csv(f"{BASE_DIR}/code/openscene/dataset/matterport/category_mapping.tsv", delimiter="\t")
print(df)

uniques = pd.unique(df["mpcat40"])
print(uniques)
names = []
counts = []
for unique in uniques:
    t = df[df["mpcat40"] == unique]
    s = t["count"].sum()
    counts.append(s)
    names.append(unique)

counts, names = np.array(counts), np.array(names)
idx = np.argsort(counts)
counts, names = counts[idx], names[idx]
print(np.stack((counts, names)).T)
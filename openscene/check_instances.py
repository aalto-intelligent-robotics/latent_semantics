import argparse
from tqdm import tqdm
import numpy as np
import glob
import os

mapids = {
    '5LpN3gDmAk7': 0,
    'gTV8FGcVJC9': 1,
    'jh4fc5c5qoQ': 2,
    'JmbYfDe2QKZ': 3,
    'mJXqzFtmKg4': 5,
    'ur6pFq6Qu1A': 6,
    'UwV83HsGsw3': 7,
    'Vt2qJdWjCF2': 8,
    'YmJkqBEsHnH': 9,
}

def checkSemantics(regions_, grid_):
    output = True
    uq_regions = np.unique(regions_)
    # for region in tqdm(uq_regions, leave=False, desc="check semantics"):
    c = 0
    l = len(uq_regions)

    #print(l, "regions to check")
    lin_grid = grid_.reshape(-1)
    lin_regions = regions_.reshape(-1)
    # for region in uq_regions:
    for i in range(len(uq_regions)):
        region = uq_regions[i]
        # if (c % 100 == 0):
        #     p = c/l*100
        #     print(p, "%")
        # c += 1

        idx = (lin_regions == region)
        sem = lin_grid[idx]
        # sem = np.take(grid_, idx)
        # us, cs = np.unique(sem, return_counts=True)
        us, cs = np.unique(sem, return_counts=True)

        if (us.size > 1 and region >= 0):
            print(str(region) + ": MIXED SEMANTICS!!!!!")
            output = False
        # else:
            # print(str(region) + ": all ok")
    return output

def getMapId(str):
    return mapids[str]

def group_files_by_vars(folder_path):
    grouped_files = {}
    unique_var1 = set()
    unique_var2 = set()

    # Iterate over all .npy files in the folder
    for file_path in sorted(glob.glob(os.path.join(folder_path, "*.npy"))):
        filename = os.path.basename(file_path)

        # Extract VAR1 and VAR2 from the filename
        # Assuming filename format: VAR1_regionX_openscene_feat_VAR2.npy
        parts = filename.split('_')
        if len(parts) == 5:
            var1 = parts[0]
            var2 = parts[4].split('.')[0]  # Remove the file extension

            # Add to unique sets
            unique_var1.add(var1)
            unique_var2.add(var2)

            # Group files by VAR1 and VAR2
            key = (var1, var2)
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(file_path)

    return grouped_files, list(unique_var1), list(unique_var2)

def main():
    parser = argparse.ArgumentParser("./matterport_conversion.py")
    parser.add_argument(
        '--input', '-i',
        dest="input",
        type=str,
        required=False,
        default="instances-lseg"
        #default="instances-openseg"
    )
    FLAGS, unparsed = parser.parse_known_args()
    input = FLAGS.input

    outs = []

    files = sorted(glob.glob(os.path.join(input, '*.npy')))
    for file in files:
        d = np.load(file)
        gt = d[0,:]
        pred = d[1,:]
        instances = d[5,:]

        out = checkSemantics(instances, gt)
        outs.append(out)
    outs = np.array(outs)

    if(np.all(outs)):
        print("all ok")


if __name__ == '__main__':
    main()
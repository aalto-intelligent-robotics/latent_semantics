import argparse
from tqdm import tqdm
import numpy as np
import glob
import os
import torch
from unique_checker import UniqueChecker

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
    parser = argparse.ArgumentParser("./check_instance_labels.py")
    parser.add_argument(
        '--input', '-i',
        dest="input",
        type=str,
        required=False,
        default="data/matterport_3d/train"
    )
    FLAGS, unparsed = parser.parse_known_args()
    input = FLAGS.input

    sem = UniqueChecker()

    path = input
    files = sorted(glob.glob(os.path.join(path, '*.pth')))

    for file in tqdm(files, leave=False):
        #    torch.save((coords, colors, vertex_labels), os.path.join(out_dir,  scene_name+'_' + region_name + '.pth'))
        (coords, colors, vertex_labels) = torch.load(file)
        sem.add(vertex_labels)

    sem.eval()
    sem.print()


if __name__ == '__main__':
    main()

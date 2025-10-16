import numpy as np
import argparse
import glob
import os
from pprint import pprint
from tqdm import tqdm
import os

from utils.classification import classify
import utils.common as common
import utils.result_writing as results
from joblib import Parallel, delayed

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
    for file_path in glob.glob(os.path.join(folder_path, "*.npy")):
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


# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


if __name__ == '__main__':

    BASE_DIR = os.environ.get("BASE_DIR")
    parser = argparse.ArgumentParser("./matterport_conversion.py")
    parser.add_argument(
        '--input', '-i',
        dest="input",
        type=str,
        required=False,
        default="save-openseg"
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        type=str,
        required=False,
        default=f"{BASE_DIR}code/openscene/results",
    )
    parser.add_argument(
        "--classes",
        "-c",
        dest="classes_path",
        type=str,
        required=False,
        default=f"{BASE_DIR}code/vlmaps/cfg/mpcat40.tsv",
    )
    parser.add_argument(
        '--delimiter', '-dlm',
        dest="delimiter",
        type=str,
        required=False,
        default="\t"
    )
    parser.add_argument(
        '--suffix', '-sx',
        dest="suffix",
        type=str,
        required=False,
        default="openseg"
    )
    parser.add_argument(
        '--multiprocess', '-mp',
        dest="multiprocess",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        '--label_from', '-lf',
        dest="label_from",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        '--label_to', '-lt',
        dest="label_to",
        type=int,
        required=False,
        default=42
    )
    FLAGS, unparsed = parser.parse_known_args()
    input = FLAGS.input
    output = FLAGS.output
    classes_path = FLAGS.classes_path
    delimiter = FLAGS.delimiter
    suffix = FLAGS.suffix
    multiprocess = FLAGS.multiprocess
    label_from = FLAGS.label_from
    label_to = FLAGS.label_to

# classes, labels, names = common.parseClassFile(classes_path, delimiter)
labels = range(label_from, label_to+1)

grouped_files, maps, types = group_files_by_vars(input)

cl_out_file = results.getFile(output, "queryability-" + suffix, "csv")
results.writeClassificationHeader(cl_out_file)
cl_dbg_file = results.getFile(output, "queryability-debug-" + suffix, "csv")
header = "map;id;tp;tn;fp;fn;i;u"
results.writeString(cl_dbg_file, header)

# for map in tqdm(maps, desc="maps"):
for mapname in tqdm(mapids, desc="maps"):
    map = mapname
    predictions = []
    for maptype in tqdm(types, desc="types", leave=False):
        first = True
        files = grouped_files[(map, maptype)]
        for file in files:
            d = np.load(file)

            if(first):
                gt = d[0,:]
                pred = d[1,:]
                first = False
            else:
                gt_temp = d[0, :]
                pred_temp = d[1, :]
                gt = np.concatenate((gt, gt_temp))
                pred = np.concatenate((pred, pred_temp))
        predictions.append((pred, maptype))
    mapId = getMapId(map)
    out, classifications = classify(predictions, gt, labels, mapId, True, multiprocess)
    results.writeClassificationLine(cl_out_file, out)

    for cls in classifications:
        line = str(map) + ";" + cls.tostring()
        results.writeString(cl_dbg_file, line)

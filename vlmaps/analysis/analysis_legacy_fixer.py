import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def parseClassFile(file, delim):
    f = open(file, 'r')
    lines = f.readlines()
    first = True

    numLines = len(lines)
    labels = np.zeros((numLines-1), dtype=np.int32)
    names = []
    idx = 0
    for line in lines:
        if (first):
            first = False
            continue

        parts = line.split(delim)
        labels[idx] = parts[0]
        names.append(parts[1].rstrip("\n").lstrip(" ").rstrip(" "))
        idx += 1
    classes = np.array((labels, names))
    f.close()

    indices = np.argsort(classes[0, :].astype(np.int32))
    classes = classes[:, indices]

    return classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./analysis_legacy_fixer.py")
    parser.add_argument(
        '--file', '-f',
        dest="file",
        type=str,
        required=False,
        default=os.environ['VLMAPS_DIR'] + "/data/mapdata/analysis/stats-bug.out"
    )
    parser.add_argument(
        '--classes', '-c',
        dest="classes",
        type=str,
        required=False,
        default=os.environ['VLMAPS_DIR'] + "/cfg/mpcat40.tsv"
    )
    parser.add_argument(
        '--delimiter', '-dlm',
        dest="delimiter",
        type=str,
        required=False,
        default="\t"
    )
    parser.add_argument(
        '--out', '-o',
        dest="out",
        type=str,
        required=False,
        default=os.environ['VLMAPS_DIR'] + "/data/mapdata/analysis/stats-fixed.out"
    )
    parser.add_argument(
        '--type', '-t',
        dest="type",
        type=int,
        required=False,
        default=0
    )

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    file = FLAGS.file
    classes = FLAGS.classes
    delimiter = FLAGS.delimiter
    out = FLAGS.out
    type = FLAGS.type

    # fix stats.out
    if(type == 0):
        classes = parseClassFile(classes, delimiter)
        # print(classes.T)
        df = pd.read_csv(file, delimiter="; ", engine="python")

        updated = df.copy()

        rows = df.shape[0]
        for r in tqdm(range(rows)):
            row = df.iloc[r]
            qn = row["queryName"]
            ql = row["queryLabel"]
            i = np.where(classes[0,:].astype(dtype=np.int32) == ql)
            idx = i[0].item()
            label = classes[0, idx]
            name = classes[1, idx]
            updated.loc[r, ["queryName"]] = name

        #updated.to_csv(out, sep=";", na_rep="n/a", index=False)
        updated.to_csv(out, sep=";", na_rep="", index=False)
        exit()
    else:
        df = pd.read_csv(file, delimiter="; ", engine="python")
        df.to_csv(out, sep=";", na_rep="", index=False)
        exit()


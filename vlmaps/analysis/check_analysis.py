import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
try:
    from utils.logger import Logger, LogLevel
except:
    import sys
    sys.path.append("utils")
    from logger import Logger, LogLevel

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./check_analysis.py")
    parser.add_argument(
        '--dir', '-f',
        dest="dir",
        type=str,
        required=False,
        default=os.environ['VLMAPS_DIR'] + "/data/iout/analysis"
    )
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dir = FLAGS.dir
    logger = Logger("check_analysis")

    # BATCH CHECK
    logger.info("Checking batch...")
    # Read File
    batch = dir + "/" + "batch.out"
    bdf = pd.read_csv(batch, delimiter="; ", engine="python")

    # get classes and maps
    classes = pd.unique(bdf["name"])
    maps = pd.unique(bdf["map"])
    numClasses = classes.size
    numMaps = maps.size
    checks = []

    # check combinations
    for cls in classes:
        cquery = bdf[bdf["query"] == cls]
        for map in maps:
            mquery = cquery[cquery["map"] == map]["name"]
            checks.append((mquery == classes).all())
    arr = np.array(checks)
    allCheck = np.all(arr)

    #output
    logger.info("Found", numMaps, "maps with", numClasses, "classes")
    logger.info("Batch check:", allCheck)

    # STATS CHECK
    stats = dir + "/" + "stats.out"
    sdf = pd.read_csv(stats, delimiter="; ", engine="python")
    labels = pd.unique(sdf["label"])
    q_labels = pd.unique(sdf["queryLabel"])

    checks = []
    for qmap in tqdm(maps, leave=False):
        for qlabel in tqdm(q_labels, leave=False):
            comps = sdf[(sdf["queryLabel"] == qlabel)
                        & (sdf["queryMap"] == qmap)]
            for map in maps:
                for label in labels:
                    item = comps[(comps["map"] == map) & (comps["label"] == label)]
                    checks.append(item.shape[0] == 1)
    arr = np.array(checks)
    allCheck = np.all(arr)
    logger.info("Stats check:", allCheck)


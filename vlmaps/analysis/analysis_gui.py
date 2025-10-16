import argparse
import os
import time
from tqdm import tqdm
import numpy as np
from pprint import pprint
import pickle
import datetime
from os import listdir, makedirs
from os.path import isfile, join, exists
import pandas
from pandasgui import show

###########################################
# MAIN
###########################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./analysis_gui.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--file', '-f',
        dest="file",
        type=str,
    )

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    file = FLAGS.file

    df = pandas.read_csv(file, delimiter=";", engine="python")
    show(df)

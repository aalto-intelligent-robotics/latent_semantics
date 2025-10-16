from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.colors import Normalize
import os

def normalize(M):
    N = (M - np.min(M)) / (np.max(M) - np.min(M))
    return N

if __name__ == '__main__':
    BASE_DIR = os.environ.get("BASE_DIR")
    parser = argparse.ArgumentParser("./matterport_conversion.py")
    parser.add_argument(
        '--images', '-i',
        dest="images",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/rgb"
    )
    parser.add_argument(
        '--depth', '-d',
        dest="depth",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/depth"
    )
    parser.add_argument(
        '--labels', '-l',
        dest="labels",
        type=str,
        required=False,
        #default=os.environ['DATA_DIR'] + "/vlmaps_dataset/5LpN3gDmAk7_1/semantic-rt"
        default=f"{BASE_DIR}/Desktop/test/semantic-rt"
    )
    FLAGS, unparsed = parser.parse_known_args()
    images = FLAGS.images
    labels = FLAGS.labels
    depths= FLAGS.depth

img_files = sorted([f for f in listdir(images) if isfile(join(images, f))])
label_files = sorted([f for f in listdir(labels) if isfile(join(labels, f))])
depth_files = sorted([f for f in listdir(depths) if isfile(join(depths, f))])

idx = 0
inp = ''
while (inp != 'q'):
    img = mpimg.imread(images+"/"+img_files[idx])
    label = np.load(labels+"/"+label_files[idx])
    depth = np.load(depths+"/"+depth_files[idx])
    nlabel = normalize(label)
    ndepth = normalize(depth)
    clabel = cm.viridis(nlabel)
    crgb = img + clabel[:, :, 0:3]

    cdepth = np.stack((ndepth, nlabel, np.zeros_like(label))).transpose(1, 2, 0)
    # cdepth = np.stack((normalize(depth), np.zeros_like(label),
    #                   np.zeros_like(label))).transpose(1, 2, 0)
    # comb[:, :, 0] = np.multiply(img[:, :, 0], label)
    # comb[:, :, 1] = np.multiply(img[:, :, 1], label)
    # comb[:, :, 2] = np.multiply(img[:, :, 2], label)

    print(np.min(label))
    print(np.max(label))
    print(img.shape)

    fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(img)
    ax2.imshow(label)
    ax3.imshow(depth)
    ax4.imshow(crgb)

    ax6.imshow(cdepth)
    plt.show()

    inp = input(": ")
    if (inp == 'a'):
        idx -= 1
    elif (inp == 'd'):
        idx += 1
    elif (inp.isdigit()):
        idx = int(inp)
    if (idx < 0):
        idx == 0
    if (idx > len(img_files)):
        idx = len(img_files)
exit()

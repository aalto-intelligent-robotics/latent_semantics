import numpy as np
import argparse
import glob
import os
from pprint import pprint
from tqdm import tqdm
import open3d as o3d

from utils import instance_segmentation
from utils import common
from joblib import Parallel, delayed

def task(file_path):
    print("read", file_path)
    d = np.load(file_path)
    gt = d[0,:]
    pred = d[1,:]
    grid = d[2:5,:]

    grid = grid.T

    filename = os.path.basename(file_path)
    outpath = os.path.join(output, filename)
    directory_path = os.path.dirname(outpath)
    file_exists = os.path.exists(outpath)

    if(file_exists and not overwrite):
        print(outpath, "already exists")
        return

    grow_range = 5 #5
    join_range = 5 #5
    instances = instance_segmentation.instance_segmentation(gt, grid, instanceMinSize, grow_range, join_range)
    outd = np.stack((gt, pred, grid[:, 0], grid[:, 1], grid[:, 2], instances))

    os.makedirs(directory_path, exist_ok=True)
    np.save(outpath, outd)

    # print("unique instances", np.unique(instances).size, np.unique(instances))

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(grid)
    # colors = common.color_instances(instances, True)
    # colors = colors/255
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])


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
        default=f"{BASE_DIR}/code/openscene/save-openseg"
    )
    parser.add_argument(
        '--output', '-o',
        dest="output",
        type=str,
        required=False,
        default=f"{BASE_DIR}/code/openscene/instances-openseg-2"
    )
    #! NOTE THAT THIS HAS TO BE THE SAME THAN IN vlmaps/config/instance_segmentation_cfg.yaml
    parser.add_argument(
        '--instanceMinSize', '-ims',
        dest="instanceMinSize",
        type=int,
        required=False,
        default=10
    )
    parser.add_argument(
        '--overwrite', '-w',
        dest="overwrite",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        '--multiprocess', '-mp',
        dest="multiprocess",
        type=int,
        required=False,
        default=0
    )

    FLAGS, unparsed = parser.parse_known_args()
    input = FLAGS.input
    output = FLAGS.output
    instanceMinSize = FLAGS.instanceMinSize
    overwrite = FLAGS.overwrite
    multiprocess = FLAGS.multiprocess

    paths = sorted(glob.glob(os.path.join(input, '*.npy')))
    if(multiprocess == 0):
        for file_path in tqdm(paths, leave=False):
            task(file_path)
    else:
        Parallel(n_jobs=multiprocess)(delayed(task)(file_path) for file_path in tqdm(paths, leave=False))

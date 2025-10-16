import glob, os
import multiprocessing as mp
import numpy as np
import plyfile
import torch
import pandas as pd
from pprint import pprint

VLMAPS_LABELS = ['wall',
'floor',
'chair',
'door',
'table',
'picture',
'cabinet',
'pillow',
'window',
'sofa',
'bed',
'curtain',
'night stand',
'plant',
'sink',
'stairs',
'ceiling',
'toilet',
'stool',
'towel',
'mirror',
'television',
'shower',
'column',
'bathtub',
'counter',
'fireplace',
'lamp',
'beam',
'banister',
'shelves',
'blinds',
'excercise equipment',
'bench',
'board',
'furniture',
'appliances',
'clothes',
'objects',
'misc']

def process_one_scene(fn):
    '''process one scene.'''

    scene_name = fn.split('/')[-3]
    region_name = fn.split('/')[-1].split('.')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1

    category_id = a['face']['category_id']
    category_id[category_id==-1] = 0
    mapped_labels = mapping[category_id]

    triangles = a['face']['vertex_indices']
    vertex_labels = np.zeros((coords.shape[0], num_classes+1), dtype=np.int32)
    # calculate per-vertex labels
    for row_id in range(triangles.shape[0]):
        for i in range(3):
            vertex_labels[triangles[row_id][i],
                            mapped_labels[row_id]] += 1

    vertex_labels = np.argmax(vertex_labels, axis=1)
    vertex_labels[vertex_labels==0] = 256
    vertex_labels -= 1

    torch.save((coords, colors, vertex_labels),
            os.path.join(out_dir,  scene_name+'_' + region_name + '.pth'))
    print(fn)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

BASE_DIR = os.environ.get("BASE_DIR")
#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'test' # 'train' | 'val' | 'test'
num_classes = 42 # 40 | 80 | 160 # define the number of classes
out_dir = '../../data/matterport_3d_40/{}'.format(split)
matterport_path = f'{BASE_DIR}/hdd/datasets/matterport3d/v1/scans'  # downloaded original matterport data
tsv_file = '../../dataset/matterport/category_mapping.tsv'
scene_list = process_txt('../../dataset/matterport/scenes_{}.txt'.format(split))
#####################################

os.makedirs(out_dir, exist_ok=True)
category_mapping = pd.read_csv(tsv_file, sep='\t', header=0)

# obtain label mapping for new number of classes
label_name = []
label_id = []
# label_all = category_mapping['nyuClass'].tolist()
label_all = category_mapping['mpcat40index'].tolist()
eliminated_list = ['void', 'unknown']
mapping = np.zeros(len(label_all)+1, dtype=int) # mapping from category id
instance_count = category_mapping['count'].tolist()
ins_count_list = []
counter = 1
flag_stop = False
# for i, x in enumerate(label_all):
#     if not flag_stop and isinstance(x, str) and x not in label_name and x not in eliminated_list:
#         label_name.append(x)
#         label_id.append(counter)
#         print(x, counter)
#         mapping[i+1] = counter
#         counter += 1
#         ins_count_list.append(instance_count[i])
#         if counter == num_classes+1:
#             flag_stop = True
#     elif isinstance(x, str) and x in label_name:
#         # find the index of the previously appeared object name
#         mapping[i+1] = label_name.index(x)+1

# print(VLMAPS_LABELS)

for i, x in enumerate(label_all):
    mapping[i+1] = x

# idx = 0
# for val in mapping:
#     print(idx, val)
#     idx += 1

# #debug
# for label in VLMAPS_LABELS:
#     idx = VLMAPS_LABELS.index(label)+1
#     find = np.where(mapping == idx)[0].size
#     print("label", label, idx, "found", find, "times")

files = []
for scene in scene_list:
    files = files + glob.glob(os.path.join(matterport_path, scene, 'region_segmentations', '*.ply'))

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_scene, files)
p.close()
p.join()

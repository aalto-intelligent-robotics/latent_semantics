
import matplotlib.pyplot as plt
from pprint import pprint
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import argparse
from tqdm import tqdm
import random
from pathlib import Path
try:
    from embeddings_from_images.pixelEmbeddingCreator import PixelEmbeddingCreator
except:
    import sys
    sys.path.append("embeddings_from_images")
    from pixelEmbeddingCreator import PixelEmbeddingCreator

def loadDataSet(img_dir, annotation_file):
    # dataset loading
    coco = COCO(annotation_file)
    files = [f for f in tqdm(listdir(img_dir), desc="read images", leave=False) if isfile(join(img_dir, f))]

    # Category IDs.
    cat_ids = coco.getCatIds()

    # All categories.
    cats = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]

    return coco, files, cat_ids, cats, cat_names

def shuffle(list):
    return sorted(list, key=lambda x: random.random())

def getId(annotation):
    return annotation["category_id"]

def getName(cat_names, annotation):
    cat_id = getId(annotation)
    idx = cat_ids.index(cat_id)
    return cat_names[idx]

def getIdFromName(cat_names, cat_ids, name):
    idx = cat_names.index(name)
    return cat_ids[idx]

def getImagesForClasses(files, classes, numToFind):
    files = shuffle(files)
    numFound = {}
    foundPerCls = {}
    foundPerClsDBG = {}
    found = []
    for ctf in classes:
        numFound[ctf] = 0
        foundPerCls[ctf] = []
        foundPerClsDBG[ctf] = []


    foundAll = False
    for file in tqdm(files, leave=False):
        image_id = getIdFromFile(file)
        img = coco.imgs[image_id]
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        for ann in anns:
            name = getName(cat_names, ann)
            if name in classes and numFound[name] < numToFind:
                numFound[name] = numFound[name] + 1
                foundPerCls[name].append(image_id)
                foundPerClsDBG[name].append(name)
                found.append(image_id)
                break

        all = True
        for k in numFound:
            v = numFound[k]
            if v < numToFind:
                all = False
                break
        if (all):
            print("found all!")
            #pprint(numFound)
            foundAll = True
            break

    if (not foundAll):
        print("Couldn't find all, but went thgouth the dataset. This is what was found")
        pprint(numFound)

    return found, foundPerCls, numFound, foundAll, foundPerClsDBG

def getIdFromFile(filename):
    #filename = files[num]
    filename = filename.replace(".jpg", "")
    return int(filename)


def createOutArray(embeddings, annotation, embedding_size):
    n = embeddings.shape[0]
    array = np.zeros((n, 2, embedding_size))
    array[:, 0, 0] = getId(annotation)
    array[:, 1, :] = embeddings
    return array

def getClassIdFromImage(dict, toFind):
    for key in dict:
        if toFind in dict[key]:
            return key
    return ""

def mixIntoMaps(classes):
    maps = {}
    labels = []
    for k in classes:
        maps[k] = []
        labels.append(k)

    midx = 0
    for k in classes:
        imgs = classes[k]
        for img in imgs:
            cls = labels[midx]
            maps[cls].append(img)
            midx += 1
            if(midx >= len(labels)):
                midx = 0
    return maps

###########################################
# MAIN
###########################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./analysis_tool.py")
    parser.add_argument(
        '--img_dir', '-i',
        dest="img_dir",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + '/coco/train2017'
    )
    parser.add_argument(
        '--annotation_file', '-a',
        dest="annotation_file",
        type=str,
        required=False,
        default=os.environ['DATA_DIR'] + '/coco/annotations/instances_train2017.json'
    )
    parser.add_argument(
        '--num', '-n',
        dest="num",
        type=int,
        required=False,
        default=100
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Use cpu or cuda.",
    )
    parser.add_argument(
        "--crop_size",
        dest="crop_size",
        type=int,
        help="crop_size of the images",
        required=False,
        default=352
    )
    parser.add_argument(
        "--weights_path",
        dest="weights_path",
        type=str,
        help="Path to the LSeg model weights.",
        required=False,
        default="models/demo_e200.ckpt"
    )
    parser.add_argument(
        "--out_path",
        dest="out_path",
        type=str,
        help="Path of output",
        required=False,
        default="data/iout/image_data"
    )
    parser.add_argument(
        "--embedding_size",
        dest="embedding_size",
        type=int,
        help="Size of the embedding",
        required=False,
        default=512
    )
    parser.add_argument(
        "--subsample",
        dest="subsample",
        type=int,
        help="Subsampling",
        required=False,
        default=5
    )
    parser.add_argument(
        "--cfg_path",
        dest="cfg_path",
        type=str,
        help="cfg_path",
        required=False,
        default="cfg/image_cats.cfg"
    )
    parser.add_argument(
        "--search_cats",
        dest="search_cats",
        type=str,
        help="search_cats",
        required=False,
        default="image_gt_generator/search_cats.cfg"
    )
    parser.add_argument(
        "--split",
        dest="split",
        action="store_true",
        help="split",
        required=False,
        default=False
    )
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    img_dir = FLAGS.img_dir
    annotation_file = FLAGS.annotation_file
    num = FLAGS.num
    device = FLAGS.device
    crop_size = FLAGS.crop_size
    weights_path = FLAGS.weights_path
    out_path = FLAGS.out_path
    embedding_size = FLAGS.embedding_size
    subsample = FLAGS.subsample
    cfg_path = FLAGS.cfg_path
    search_cats = FLAGS.search_cats
    split = FLAGS.split

    # embedding creator
    creator = PixelEmbeddingCreator(weights_path, crop_size)

    # load dataset
    coco, files, cat_ids, cats, cat_names = loadDataSet(img_dir, annotation_file)

    # find a sample set of images
    f = open(search_cats, "r")
    cats_to_find = []
    for line in f.readlines():
        cats_to_find.append(line.rstrip("\n").rstrip(" ").lstrip(" "))
    found, foundPerCls, numFound, foundAll, foundPerClsDBG = getImagesForClasses(
        files, cats_to_find, num)

    if(num < len(cats_to_find) and split):
        print("*"*80)
        print("Warning! Less samples than classes to sample from. This leads to unbalanced maps")
        print("*"*80)

    if (split):
        mapSplit = mixIntoMaps(foundPerCls)
    #mapSplitDBG = mixIntoMaps(foundPerClsDBG)

    allcats = []
    allids = []
    outInit = False

    # ids = np.zeros((len(found),1))
    # idx = 0
    # for image_id in tqdm(found, leave=False):
    #     map_id = getClassIdFromImage(foundPerCls, image_id)
    #     ids[idx] = map_id
    #     idx += 1
    #     print(image_id, map_id)
    # print(np.unique(ids))
    allEmbeddings = {}
    outInit = {}
    for cat in cats_to_find:
        outInit[cat] = False

    for image_id in tqdm(found, leave=False):
        try:
            # print("*"*80)
            # print("id", image_id)
            # example
            img = coco.imgs[image_id]
            image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))

            if(image.shape[1] < crop_size):
                eff_crop_size = crop_size / 2
            else:
                eff_crop_size = crop_size
            base_size = image.shape[1]
            image_embeddings = creator.get_embeddings(
                image, eff_crop_size, base_size)
            #print("Embeddings:", image_embeddings.shape, image_embeddings.dtype)

            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
        except:
            continue

        map_id = 0
        if(split):
            # pprint(mapSplit)
            map_id = getClassIdFromImage(mapSplit, image_id)
            # print(image_id, "in", map_id)

        for ann in tqdm(anns, leave=False):
            try:
                # get embeddings
                mask = coco.annToMask(ann)
                masksum = np.sum(mask)
                mask = np.array(mask, dtype=np.bool_)
                seg_embeddings = image_embeddings[mask,:]
                seg_embeddings = seg_embeddings[::subsample]

                if not outInit[map_id]:
                    allEmbeddings[map_id] = createOutArray(seg_embeddings, ann, embedding_size)
                    outInit[map_id] = True
                else:
                    current = createOutArray(seg_embeddings, ann, embedding_size)
                    allEmbeddings[map_id] = np.concatenate((current, allEmbeddings[map_id]), axis=0)

                # for k in allEmbeddings:
                #     print("k:", k, outInit[k], len(allEmbeddings[k].shape))

                # store all classes
                name = getName(cat_names, ann)
                if (not name in allcats):
                    allcats.append(name)
                    id = getIdFromName(cat_names, cat_ids, name)
                    allids.append(id)
            except:
                continue

    out_dir = os.path.dirname(os.path.abspath(out_path))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    i = 0
    for cat in allEmbeddings:
        np.save(out_path+"-"+str(i), allEmbeddings[cat])
        i += 1
    print("Data written to", out_path)

    cfg = np.array((allids, allcats))
    print(cfg.shape)
    indices = np.argsort(cfg[0, :].astype(np.int32))
    print(indices.shape)
    cfg = cfg[:, indices]

    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    Path(cfg_dir).mkdir(parents=True, exist_ok=True)
    f = open(cfg_path, "w")
    f.write("id; name\n")
    for c in range(cfg.shape[1]):
        id = cfg[0, c]
        name = cfg[1, c]
        f.write(str(id) + "; " + name + "\n")
    f.close()

    print("Configuration written to", cfg_path)
    print("Done")
import argparse
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import numpy as np
import utils.common as common
import dataset.label_constants as lc
from pprint import pprint
import matplotlib.pyplot as plt
import os
from unique_checker import UniqueChecker

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def printArr(data):
    cs = consecutive(data, 1)
    first = True

    printstr = ""
    for c in cs:
        if(not first):
            printstr += ", "
        if(c.size > 1):
            printstr += str(c[0]) + " - " + str(c[-1])
        else:
            printstr += str(c[0])
        first = False
    print(printstr)

def find_files_with_string(folder_path, search_string):
    found_files = []
    # Iterate through all files and directories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the search string is present in the filename
            if search_string in file:
                found_files.append(os.path.join(root, file))
    return found_files

def main():
    parser = argparse.ArgumentParser("./show_map.py")
    parser.add_argument(
        '--input', '-i',
        dest="file_path",
        type=str,
        required=False,
        default=""
    )
    parser.add_argument(
        '--dir', '-d',
        dest="dir",
        type=str,
        required=False,
        #default="mp40-evaluation/eval-openseg/result_eval"
        #default="mp40-evaluation/eval-lseg/result_eval"
        #default="mp40-evaluation/instances-openseg"
        #default="mp40-evaluation/instances-lseg"
        default="mp40-evaluation/save-openseg"
        #default="mp40-evaluation/save-lseg"
    )
    parser.add_argument(
        '--color', '-c',
        dest="color",
        type=str,
        required=False,
        default="i"
    )
    parser.add_argument(
        '--maps', '-m',
        dest="maps",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        '--verbose', '-v',
        dest="verbose",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        '--contrastive', '-x',
        dest="contrastive",
        action="store_true",
        required=False,
        default=False
    )
    FLAGS, unparsed = parser.parse_known_args()
    file_path = FLAGS.file_path
    dir = FLAGS.dir
    color = FLAGS.color
    maps = FLAGS.maps
    verbose = FLAGS.verbose
    contrastive = FLAGS.contrastive

    # cmap = lc.MATTERPORT_COLOR_MAP_21
    # labels = lc.MATTERPORT_LABELS_21
    cmap = lc.MATTERPORT_COLOR_MAP_160
    labels = lc.VLMAPS_LABELS

    # print("labels:")
    # pprint(labels)
    # print("cmap:")
    # pprint(cmap)


    if(contrastive):
        file_path = getFile(dir)
        file_path2 = getFile(dir)
        if(file_path != file_path2):
            cloud1 = o3d.io.read_point_cloud(file_path)
            cloud2 = o3d.io.read_point_cloud(file_path2)


            cp1 = np.asarray(cloud1.points)
            print(cp1.shape)
            cc1 = np.asarray(cloud1.colors)
            print(cc1.shape)
            cp2 = np.asarray(cloud2.points)
            print(cp2.shape)
            cc2 = np.asarray(cloud2.colors)
            print(cc2.shape)

            newcols = np.full_like(cc1, 0)
            for i in range(cc1.shape[0]):
                col1 = cc1[i,:]
                col2 = cc2[i,:]
                if np.all(col1 == col2):
                    newcol = (0,0.5,0)
                else:
                    newcol = (1,0,0)
                newcols[i,:] = newcol

            cloud1.colors = o3d.utility.Vector3dVector(newcols)
            o3d.visualization.draw_geometries([cloud1])
            exit()

    hasInstances = False

    if (not file_path):
        file_path = getFile(dir)

    # ply
    if ("ply" in file_path):
        cloud = o3d.io.read_point_cloud(file_path)
        o3d.visualization.draw_geometries([cloud])
    # npy mode
    elif ("npy" in file_path):
        predc = UniqueChecker("Predicted")
        gtc = UniqueChecker("GT")

        if(maps):
            path, file_name = os.path.split(file_path)
            mapname = file_name.split("_")[0]
            files = find_files_with_string(path, mapname)
            first = True
            file_i = 0
            for file in files:
                d = np.load(file)
                if(first):
                    gt = d[0, :]
                    pred = d[1, :]
                    grid = d[2:5, :]
                    if(d.shape[0] > 5):
                        instances = d[5, :]
                        hasInstances = True
                    first = False
                else:
                    gt_temp = d[0, :]
                    pred_temp = d[1, :]
                    grid_temp = d[2:5, :]
                    gt = np.concatenate((gt, gt_temp))
                    pred = np.concatenate((pred, pred_temp))
                    grid = np.concatenate((grid, grid_temp),axis=1)
                    if(hasInstances):
                        instances_temp = d[5, :]
                        instances_temp[instances_temp >= 0] = instances_temp[instances_temp >= 0] + (file_i * 1000)
                        instances = np.concatenate((instances, instances_temp))
                file_i += 1

                predc.add(pred)
                gtc.add(gt)
        else:
            d = np.load(file_path)
            gt = d[0, :]
            pred = d[1, :]
            grid = d[2:5, :]
            print(d.shape)
            if(d.shape[0] > 5):
                instances = d[5, :]
                hasInstances = True

            predc.add(pred)
            gtc.add(gt)

        ########################################################################
        # STATISTICS
        ########################################################################

        predc.eval()
        predc.print()
        gtc.eval()
        gtc.print()

        #####################################
        # GT
        #####################################
        print(" ")
        print(" ")
        print(" ")
        print("GT")
        print("*"*80)

        unique, count = np.unique(gt, return_counts=True)
        idx = np.flip(np.argsort(count))
        unique = unique[idx]
        count = count[idx]

        # header
        uh = "label"
        nameh = "name"
        ch = "count"
        colh = "color"
        print(f'{uh:<5} {nameh:<20} {ch:<10} {colh}')
        print("-"*50)

        for (u, c) in zip(unique, count):
            try:
                pred_name = labels[u]
            except:
                pred_name = "* exception"
            col = common.color_semantics_single(u)
            print(f'{u:<5} {pred_name:<20} {c:<10} {col}')

        print(" ")
        printArr(np.sort(unique))

        #####################################
        # semantics
        #####################################
        print(" ")
        print(" ")
        print(" ")
        print("Semantics")
        print("*"*80)

        unique, count = np.unique(pred, return_counts=True)
        idx = np.flip(np.argsort(count))
        unique = unique[idx]
        count = count[idx]

        # header
        uh = "label"
        nameh = "name"
        ch = "count"
        colh = "color"
        print(f'{uh:<5} {nameh:<20} {ch:<10} {colh}')
        print("-"*50)

        for (u, c) in zip(unique, count):
            pred_name = labels[u]
            col = common.color_semantics_single(u)
            print(f'{u:<5} {pred_name:<20} {c:<10} {col}')

        print(" ")
        printArr(np.sort(unique))

        #####################################
        # instances summary
        #####################################
        if(hasInstances):
            gt_labels = []
            p_labels = []
            print(" ")
            print(" ")
            print(" ")
            print("Instances - summary")
            print("*"*80)

            unique, count = np.unique(instances, return_counts=True)
            idx = np.flip(np.argsort(count))
            unique = unique[idx]
            count = count[idx]

            ih = "inst."
            ch = "count"
            colh = "color"
            gnameh = "gt name"
            glh = "gt l"
            pnameh = "pr name"
            plh = "pr l"
            print(f'{ih:<5} {ch:<10} {gnameh:<10} {glh:<5} {pnameh:<10} {plh:<5} {colh:<10}')
            print("-"*60)

            for (u, c) in zip(unique, count):
                col = common.color_instances_single(u)

                vox = pred[instances == u]
                ius, ics = np.unique(vox, return_counts=True)
                idx = np.flip(np.argsort(ics))
                ius = ius[idx]
                ics = ics[idx]
                pred_label = ius[0]
                p_labels.append(pred_label)
                pred_name = labels[pred_label]

                vox = gt[instances == u]
                ius, ics = np.unique(vox, return_counts=True)
                idx = np.flip(np.argsort(ics))
                ius = ius[idx]
                ics = ics[idx]
                gt_label = ius[0]
                gt_labels.append(gt_label)
                try:
                    gt_name = labels[gt_label]
                except:
                    gt_name = "* except"
                print(f'{u:<5} {c:<10} {gt_name:<10} {gt_label:<5} {pred_name:<10} {pred_label:<5} {col}')


            arr = np.array(gt_labels)
            uq, cnt = np.unique(arr, return_counts=True)
            idx = np.flip(np.argsort(cnt))
            uq, cnt = uq[idx], cnt[idx]
            print(" ")
            print("GT")
            uh = "label"
            nameh = "name"
            ch = "count"
            print(f'{uh:<5} {nameh:<10} {ch:<5}')
            print("-"*30)
            for (u, c) in zip(uq, cnt):
                try:
                    pred_name = labels[u]
                except:
                    pred_name = "* except"
                print(f'{u:<5} {pred_name:<10} {c:<5}')
            print("     " + "---------------")
            print("     " + " total      " + str(np.sum(cnt)))

            arr = np.array(p_labels)
            uq, cnt = np.unique(arr, return_counts=True)
            idx = np.flip(np.argsort(cnt))
            uq, cnt = uq[idx], cnt[idx]
            print(" ")
            print("Predicted")
            uh = "label"
            nameh = "name"
            ch = "count"
            print(f'{uh:<5} {nameh:<10} {ch:<5}')
            print("-"*30)
            for (u, c) in zip(uq, cnt):
                pred_name = labels[u]
                print(f'{u:<5} {pred_name:<10} {c:<5}')
            print("     " + "---------------")
            print("     " + " total      " + str(np.sum(cnt)))

            #####################################
            # instances details
            #####################################
            if (verbose):
                print(" ")
                print(" ")
                print(" ")
                print("Instances - details")
                print("*"*80)

                unique, count = np.unique(instances, return_counts=True)
                idx = np.flip(np.argsort(count))
                unique = unique[idx]
                count = count[idx]

                ih = "inst."
                ch = "count"
                colh = "color"
                print(f'{ih:<5} {ch:<10} {colh:<10}')

                for (u, c) in zip(unique, count):
                    vox = pred[instances == u]
                    col = common.color_instances_single(u)
                    print(f'{u:<5} {c:<10} {col}')
                    print("-"*30)

                    ius, ics = np.unique(vox, return_counts=True)
                    idx = np.flip(np.argsort(ics))
                    ius = ius[idx]
                    ics = ics[idx]
                    for (iu, ic) in zip(ius, ics):
                        pred_name = labels[iu]
                        print(f'{iu:<5} {pred_name:<10} {ic:<10}')
                    print("="*60)

        ########################################################################
        # DISPLAY
        ########################################################################
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(grid.T)
        if(contrastive):
            colors = common.color_contrastive(gt, pred)
        else:
            if (hasInstances and color == "i"):
                colors = common.color_instances(instances, True)
            elif (color == "g"):
                colors = common.color_semantics(gt, True)
            else:
                #colors = coloring(pred, cmap)
                colors = common.color_semantics(pred, True)
        colors = colors/255
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

    else:
        print("unknown file type")

def getFile(dir):
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(initialdir=dir)
    return file


def coloring(grid, cmap):
    sem_cols = np.zeros((grid.shape[0], 3))
    for i in range(grid.shape[0]):
        label = grid[i]

        try:
            col = cmap[label]
            sem_cols[i, :] = (col[0], col[1], col[2])
        except KeyError:
            sem_cols[i, :] = (0, 0, 0)
    return sem_cols

if __name__ == '__main__':
    main()
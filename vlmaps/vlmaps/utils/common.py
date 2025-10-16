import numpy as np

#  ██████╗██╗      █████╗ ███████╗███████╗███████╗███████╗
# ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██╔════╝██╔════╝
# ██║     ██║     ███████║███████╗███████╗█████╗  ███████╗
# ██║     ██║     ██╔══██║╚════██║╚════██║██╔══╝  ╚════██║
# ╚██████╗███████╗██║  ██║███████║███████║███████╗███████║
#  ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚══════╝


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

    labels = classes[0, :]
    names = classes[1, :]

    return classes, labels, names

#  ██████╗ ██████╗ ██╗      ██████╗ ██████╗ ███████╗
# ██╔════╝██╔═══██╗██║     ██╔═══██╗██╔══██╗██╔════╝
# ██║     ██║   ██║██║     ██║   ██║██████╔╝███████╗
# ██║     ██║   ██║██║     ██║   ██║██╔══██╗╚════██║
# ╚██████╗╚██████╔╝███████╗╚██████╔╝██║  ██║███████║
#  ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝


def color_semantics(grid, onedim = False):
    sem_cols = np.zeros((grid.shape[0], 3))
    for i in range(grid.shape[0]):
        if(onedim):
            label = grid[i]
        else:
            label = grid[i, 0]
        sem_cols[i, :] = color_semantics_single(label)
    return sem_cols

def color_semantics_single(label):
    try:
        color_tuple = colormap[label]
        color = (color_tuple["r"], color_tuple["g"], color_tuple["b"])
    except KeyError:
        color = (0, 0, 0)
    return color

def color_regions(grid):
    reg_cols = np.zeros((grid.shape[0], 3))
    for i in range(grid.shape[0]):
        label = grid[i, 0]
        try:
            col = colormap[label]
            reg_cols[i, :] = (col["r"], col["g"], col["b"])
        except KeyError:
            reg_cols[i, :] = (0, 0, 0)
    return reg_cols


def color_instances(grid, onedim = False):
    print(grid.shape)
    insta_cols = np.zeros((grid.shape[0], 3))
    for i in range(grid.shape[0]):
        if(onedim):
            instance = grid[i]
        else:
            instance = grid[i, 0]

        insta_cols[i, :] = color_instances_single(instance)

        # im = instance % 255
        # insta_cols[i, :] = (im, im, im)
    return insta_cols

def color_instances_single(instance):
    idx = instance % 41
    color_tuple = colormap[idx]
    color = (color_tuple["r"], color_tuple["g"], color_tuple["b"])
    return color


colormap = {}
colormap[0] = {"r": 120, "g": 120, "b": 120}
colormap[1] = {"r": 174, "g": 199, "b": 232}
colormap[2] = {"r": 112, "g": 128, "b": 144}
colormap[3] = {"r": 152, "g": 223, "b": 138}
colormap[4] = {"r": 197, "g": 176, "b": 213}
colormap[5] = {"r": 255, "g": 127, "b": 14}
colormap[6] = {"r": 214, "g": 39, "b": 40}
colormap[7] = {"r": 31, "g": 119, "b": 180}
colormap[8] = {"r": 188, "g": 189, "b": 34}
colormap[9] = {"r": 255, "g": 152, "b": 150}
colormap[10] = {"r": 44, "g": 160, "b": 44}
colormap[11] = {"r": 227, "g": 119, "b": 194}
colormap[12] = {"r": 222, "g": 158, "b": 214}
colormap[13] = {"r": 148, "g": 103, "b": 189}
colormap[14] = {"r": 140, "g": 162, "b": 82}
colormap[15] = {"r": 132, "g": 60, "b": 57}
colormap[16] = {"r": 158, "g": 218, "b": 229}
colormap[17] = {"r": 156, "g": 158, "b": 222}
colormap[18] = {"r": 231, "g": 150, "b": 156}
colormap[19] = {"r": 99, "g": 121, "b": 57}
colormap[20] = {"r": 140, "g": 86, "b": 75}
colormap[21] = {"r": 219, "g": 219, "b": 141}
colormap[22] = {"r": 214, "g": 97, "b": 107}
colormap[23] = {"r": 206, "g": 219, "b": 156}
colormap[24] = {"r": 231, "g": 186, "b": 82}
colormap[25] = {"r": 57, "g": 59, "b": 121}
colormap[26] = {"r": 165, "g": 81, "b": 148}
colormap[27] = {"r": 173, "g": 73, "b": 74}
colormap[28] = {"r": 181, "g": 207, "b": 107}
colormap[29] = {"r": 82, "g": 84, "b": 163}
colormap[30] = {"r": 189, "g": 158, "b": 57}
colormap[31] = {"r": 196, "g": 156, "b": 148}
colormap[32] = {"r": 247, "g": 182, "b": 210}
colormap[33] = {"r": 107, "g": 110, "b": 207}
colormap[34] = {"r": 255, "g": 187, "b": 120}
colormap[35] = {"r": 199, "g": 199, "b": 199}
colormap[36] = {"r": 140, "g": 109, "b": 49}
colormap[37] = {"r": 231, "g": 203, "b": 148}
colormap[38] = {"r": 206, "g": 109, "b": 189}
colormap[39] = {"r": 23, "g": 190, "b": 207}
colormap[40] = {"r": 127, "g": 127, "b": 127}
colormap[41] = {"r": 0, "g": 0, "b": 0}

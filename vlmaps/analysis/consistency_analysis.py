import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats
import numpy as np
import warnings
from enum import Enum
import seaborn as sn
import os

# ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ ███████╗
# ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝
# ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝███████╗
# ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║
# ██║  ██║███████╗███████╗██║     ███████╗██║  ██║███████║
# ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝


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


def getNameForLabel(label, classes, names):
    idx = np.where(classes == label, True, False)
    name = names[idx]
    return name.item()

# ██████╗ ██╗      ██████╗ ████████╗███████╗
# ██╔══██╗██║     ██╔═══██╗╚══██╔══╝██╔════╝
# ██████╔╝██║     ██║   ██║   ██║   ███████╗
# ██╔═══╝ ██║     ██║   ██║   ██║   ╚════██║
# ██║     ███████╗╚██████╔╝   ██║   ███████║
# ╚═╝     ╚══════╝ ╚═════╝    ╚═╝   ╚══════╝

###########################################
# SCATTER
###########################################


class ScatterOptions(object):
    color = ""
    marker = ""
    s = 1
    alpha = 1
    linewidth = 1

    def __init__(self, color, marker, s, alpha, linewidth):
        self.color = color
        self.marker = marker
        self.s = s
        self.alpha = alpha
        self.linewidth = linewidth


def plotMainScatter(ax, matches, rest, field, metric, matchOptions: ScatterOptions, restOptions: ScatterOptions):
    ax.scatter(matches[field], matches[metric].astype(dtype=np.float32), c=matchOptions.color, s=matchOptions.s,
               marker=matchOptions.marker, linewidths=matchOptions.linewidth, alpha=matchOptions.alpha)
    ax.scatter(rest[field], rest[metric].astype(dtype=np.float32), c=restOptions.color, s=restOptions.s,
               marker=matchOptions.marker, linewidths=matchOptions.linewidth, alpha=restOptions.alpha)


def plotMeanScatter(ax, matches, rest, field, metric, items, matchOptions: ScatterOptions, restOptions: ScatterOptions):
    for item in items:
        m = matches[matches[field] == item]
        mean = np.mean(m[metric].astype(dtype=np.float32))
        ax.scatter(item, mean, s=matchOptions.s, c=matchOptions.color,
                   marker=matchOptions.marker, linewidths=matchOptions.linewidth)
        m = rest[rest[field] == item]
        mean = np.mean(m[metric].astype(dtype=np.float32))
        ax.scatter(item, mean, s=restOptions.s, c=restOptions.color,
                   marker=restOptions.marker, linewidths=restOptions.linewidth)


###########################################
# BOXPLOT
###########################################
class BoxplotOptions(object):
    def __init__(self):
        pass

class BoxplotSorting(Enum):
    means = 0,
    diff_means = 1,
    separation = 2,
    pval = 3


def plotMainBoxplot(ax, matches, rest, field, metric, sort: BoxplotSorting, smaller=False):
    items = np.unique(matches[field])

    # sort items
    running_index = 0
    sorting = []
    flip = False
    for item in items:
        d1 = matches[(matches[field] == item) & ~(matches[metric].isna())][metric].astype(dtype=np.float32)
        d2 = rest[(rest[field] == item) & ~(rest[metric].isna())][metric].astype(dtype=np.float32)
        # max mean metric
        if (sort == BoxplotSorting.means):
            m = np.mean(d1)
            flip = True
        # difference metric
        elif (sort == BoxplotSorting.diff_means):
            m1 = np.mean(d1)
            m2 = np.mean(d2)
            if (smaller):
                m = m2-m1
            else:
                m = m1-m2
            flip = True
        # maximum separation
        elif (sort == BoxplotSorting.separation):
            d1 = np.array(d1)
            d2 = np.array(d2)

            if(d1.size == 0):
                m = 0
                continue

            if(smaller):
                comp = np.max(d1)
                idx = d2 > comp
            else:
                comp = np.min(d1)
                idx = d2 < comp

            m = np.sum(idx)
        elif(sort == BoxplotSorting.pval):
            if (d1.size <= 1 or d2.size <= 1):
                m = 1
                continue

            ttype = "less" if smaller else "greater"
            res = scipy.stats.mannwhitneyu(d1, d2, alternative=ttype)
            # ttype = "less" if not smaller else "greater"
            # res = scipy.stats.ks_2samp(d1, d2, alternative=ttype)
            #res = scipy.stats.kruskal(d1, d2)
            #res = scipy.stats.ttest_ind(d1, d2, alternative=ttype)
            #res = scipy.stats.tukey_hsd(d1, d2)
            m = res.pvalue
        else:
            m = running_index

        sorting.append(m)
        running_index += 1


    sorting = np.array(sorting)
    idx = np.argsort(sorting)
    if(flip):
        idx = np.flip(idx)
    sorting = sorting[idx]
    items = items[idx]

    print(np.stack((items, sorting)).T)

    init_pos = 0
    stride = 1
    width = 0.25

    offset = -0.125
    last_pos = init_pos
    x = []
    lab = []
    pos = []
    for item in items:
        d = matches[(matches[field] == item) & ~(matches[metric].isna())][metric].astype(dtype=np.float32)
        p = last_pos + stride
        last_pos = p
        p = p + offset

        x.append(d)
        lab.append(item)
        pos.append(p)
    bp1 = ax.boxplot(x, labels=lab, patch_artist=True, positions=pos, widths=width)

    offset = 0.125
    last_pos = init_pos
    x2 = []
    lab2 = []
    pos2 = []
    for item in items:
        d2 = rest[(rest[field] == item) & ~(rest[metric].isna())][metric].astype(dtype=np.float32)
        p2 = last_pos + stride
        last_pos = p2
        p2 = p2 + offset

        x2.append(d2)
        lab2.append(item)
        pos2.append(p2)
    bp2 = ax.boxplot(x2, labels=lab2, patch_artist=True, positions=pos2, widths=width, manage_ticks=False)

    for patch in bp1['boxes']:
        patch.set_facecolor('red')

    for patch in bp2['boxes']:
        patch.set_facecolor('blue')

    ax.grid(visible=True, axis="both")

###########################################
# COMMON
###########################################


def plotDetails(ax, legend, ylabel, xlabel, title):
    ax.legend(legend, loc="lower right")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(),
                  rotation=90, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

###########################################
# METRICS
###########################################


def getMetrics(matches, rest, field, metric, items, title, max=False, verbose=False):
    print("*"*80)
    better_mean = 0
    lower_variance = 0
    best = 0
    all = 0
    for item in items:
        match = matches[matches[field] == item]
        if (match.shape[0] <= 0):
            continue
        match_val = match[metric].astype(dtype=np.float32)
        if (max):
            best_match = np.max(match_val)
        else:
            best_match = np.min(match_val)
        match_mean = np.mean(match_val)
        match_var = np.var(match_val)

        other = rest[rest[field] == item]
        other_val = other[metric].astype(dtype=np.float32)
        if (max):
            best_other = np.max(other_val)
        else:
            best_other = np.min(other_val)
        other_mean = np.mean(other_val)
        other_var = np.var(other_val)

        if (max):
            if (match_mean > other_mean):
                better_mean += 1
            if (best_match > best_other):
                best += 1
        else:
            if (match_mean < other_mean):
                better_mean += 1
            if (best_match < best_other):
                best += 1

        if (match_var < other_var):
            lower_variance += 1

        all += 1
        if verbose:
            print("="*40)
            print("item:", item)
            print("match mean:", match_mean, "var:", match_var)
            print("other mean:", other_mean, "var:", other_var)
    print("*"*80)
    print(title)
    if (max):
        print("Higher mean", better_mean, "/", all)
    else:
        print("Lower mean", better_mean, "/", all)
    print("Lower variance", lower_variance, "/", all)
    if (max):
        print("Best (max):", best, "/", all)
    else:
        print("Best (min):", best, "/", all)

########################################################################################################################

########################################################################################################################

########################################################################################################################

########################################################################################################################

########################################################################################################################

# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
if __name__ == '__main__':
    parser = argparse.ArgumentParser("./consistency_analysis.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--dir', '-d',
        dest="dir",
        type=str,
    )
    parser.add_argument(
        "--verbose", "-v",
        dest="verbose",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--classes", "-c",
        dest="classes",
        type=str,
        required=False,
        default=os.environ['VLMAPS_DIR'] + "/cfg/mpcat40_edit.tsv"
    )
    parser.add_argument(
        '--delimiter', '-dlm',
        dest="delimiter",
        type=str,
        required=False,
        default=";"
    )

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dir = FLAGS.dir
    verbose = FLAGS.verbose
    classesFile = FLAGS.classes
    delimiter = FLAGS.delimiter

    classes = parseClassFile(classesFile, delimiter)
    classlabels = classes[0, :].astype(dtype=np.int32)
    names = classes[1, :]
    clsDict = {}

    for i in range(len(classlabels)):
        label = int(classlabels[i])
        name = names[i]
        clsDict[label] = name

    queryability = False
    intermap = True
    intramap = True
    confusion = False

    sorting = BoxplotSorting.pval

    matchOptions = ScatterOptions("#ff0000", "_", 100, 1, 1)
    restOptions = ScatterOptions("#0000ff", "o", 10, 0.25, 1.5)
    matchMeanOptions = ScatterOptions("#ff0000", "_", 500, 1, 2)
    restMeanOptions = ScatterOptions("#0000ff", "_", 500, 1, 2)

    ############################################################################
    # QUERYABILITY
    ############################################################################
    if (queryability):
        # read
        batch = dir + "/" + "batch.out"
        bdf = pd.read_csv(batch, delimiter=";", engine="python")

        # OUTLIER REJECTION
        bdf = bdf[bdf["name"] != "void"]

        # queries
        matches = bdf[(bdf["name"] == bdf["query"])]
        metric = "mean"

        matches = matches.sort_values(by=metric, ascending=False)
        rest = bdf[(bdf["name"] != bdf["query"])]

        queries = pd.unique(bdf["query"])

        # plot
        fig2, q_ax = plt.subplots()
        print(" ")
        print("Queryability:")
        print("*"*60)
        plotMainBoxplot(q_ax, matches, rest, "query", metric, sorting)
        # plotMainScatter(q_ax, matches, rest, "query", metric, matchOptions, restOptions)
        # plotMeanScatter(q_ax, matches, rest, "query", metric, queries, matchMeanOptions, restMeanOptions)
        plotDetails(q_ax, ["label = query", "label != query"], "cosine distance",
                    "query", "Queryability (comparison between labels and queries)")

        # metrics
        # getMetrics(matches, rest, "name", metric, queries, "Queryability", True, False)

    ############################################################################
    # INTERMAP
    ############################################################################
    if (intermap):
        # read
        stats = dir + "/" + "stats.out"
        sdf = pd.read_csv(stats, delimiter=";", engine="python")

        # OUTLIER REJECTION
        sdf = sdf[sdf["name"] != "void"]

        # sort
        metric = "p wsd"
        if (metric == "p wsd"):
            sdf = sdf[sdf["p wsd"] != "err"]
            sdf = sdf[sdf["p wsd"] != "n/a"]

        # queries
        matches = sdf[(sdf["map"] != sdf["queryMap"]) &
                      (sdf["name"] == sdf["queryName"])]
        rest = sdf[(sdf["map"] != sdf["queryMap"]) &
                   (sdf["name"] != sdf["queryName"])]
        matches = matches.sort_values(by=metric)

        # plot
        fig, inter_ax = plt.subplots()
        print(" ")
        print("Inter:")
        print("*"*60)
        plotMainBoxplot(inter_ax, matches, rest, "name", metric, sorting, True)
        # plotMainScatter(inter_ax, matches, rest, "name", metric, matchOptions, restOptions)
        # plotMeanScatter(inter_ax, matches, rest, "name", metric, queries, matchMeanOptions, restMeanOptions)
        plotDetails(inter_ax, ["label = comp. label", "label != comp. label"], "wasserstein distance", "name",
                    "Inter-map consistency (comparison between label from map A to all labels on map B)")

        # metrics
        # getMetrics(matches, rest, "name", metric, queries, "Inter-map", False, False)

    ############################################################################
    # INTRAMAP
    ############################################################################
    if (intramap):
        # read
        instances = dir + "/" + "instance.out"
        idf = pd.read_csv(instances, delimiter=";", engine="python")

        # OUTLIER REJECTION
        idf = idf[idf["instance"] != -1]
        idf = idf[idf["queryInstance"] != -1]
        idf = idf[idf["p_wsd"] > 0]

        # add name column
        idf["name"] = idf["label"].map(clsDict)

        # queries
        matches = idf[(idf["label"] == idf["queryLabel"])]
        metric = "p_wsd"

        matches = matches.sort_values(by=metric, ascending=False)
        rest = idf[(idf["label"] != idf["queryLabel"])]

        ulabels = pd.unique(idf["label"])
        lnames = []
        for label in ulabels:
            name = getNameForLabel(label, classlabels, names)
            lnames.append(name)

        # plot
        fig2, intra_ax = plt.subplots()
        print(" ")
        print("Intra:")
        print("*"*60)
        plotMainBoxplot(intra_ax, matches, rest, "name", metric, sorting, True)
        # plotMainScatter(intra_ax, matches, rest, "name", metric, matchOptions, restOptions)
        # plotMeanScatter(intra_ax, matches, rest, "name", metric, lnames, matchMeanOptions, restMeanOptions)
        plotDetails(intra_ax, ["label = query label", "label != query label"], "wasserstein distance",
                    "label", "Intra-map consistency (comparison between instances of labels within a map)")

        # metrics
        # getMetrics(matches, rest, "name", metric, lnames, "Intra-map", False, False)

    ############################################################################
    # ANALYSIS
    ############################################################################
    # difference in means
    if(sorting == BoxplotSorting.diff_means):
        if(queryability):
            q_ax.vlines(23.5, 0.5, 1, 'r')
        if(intermap):
            inter_ax.vlines(27.5, 0, 60, 'r')
        if(intramap):
            intra_ax.vlines(14.5, 0, 60, 'r')
            intra_ax.vlines(25.5, 0, 60, 'r')
    if (sorting == BoxplotSorting.pval):
        if (queryability):
            q_ax.text(24.5, 1, "p < 0.01")
            q_ax.vlines(26.5, 0.5, 1, 'r')
            q_ax.text(27, 1, "p < 0.05")
            q_ax.vlines(28.5, 0.5, 1, 'r')
            q_ax.text(29, 1, "p >= 0.05")
        if (intermap):
            inter_ax.text(27.5, 60, "p < 0.01")
            inter_ax.vlines(29.5, 0, 60, 'r')
            inter_ax.text(30, 60, "p < 0.05")
            inter_ax.vlines(31.5, 0, 60, 'r')
            inter_ax.text(32, 60, "p >= 0.05")
        if (intramap):
            intra_ax.text(20, 60, "p < 0.01")
            #intra_ax.vlines(14.5, 0, 60, 'r')
            #intra_ax.vlines(25.5, 0, 60, 'r')

    ############################################################################
    # CONFUSION MATRIX
    ############################################################################
    if(confusion):
        confusion = dir + "/" + "confusion_matrix.out"
        cdf = pd.read_csv(confusion, delimiter=";", engine="python")

        labels_file = dir + "/" + "confusion_labels.out"
        labels = np.loadtxt(labels_file, delimiter=";", dtype=np.int32)

        lnames = []
        for label in labels:
            name = getNameForLabel(label, classlabels, names)
            lnames.append(name)

        #label rows and cols
        cdf = cdf.set_axis(lnames, axis=0)
        cdf = cdf.set_axis(lnames, axis=1)


        #fill nans and infs to 0
        cdf = cdf.replace([np.inf, -np.inf], np.nan)
        cdf = cdf.fillna(0)
        print(cdf)

        plt.figure(figsize = (10,7))
        sn.heatmap(cdf, annot=False, vmin=0, vmax=1, yticklabels=True)
        plt.xlabel("Predicted")
        plt.ylabel("True")
    ############################################################################
    # PLOT
    ############################################################################
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.show()

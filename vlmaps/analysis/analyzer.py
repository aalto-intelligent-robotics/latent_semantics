import scipy.stats
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fitter import Fitter, get_common_distributions, get_distributions
import sys
import os
from os import listdir
from os.path import isfile, join
import warnings
from sklearn import metrics
# warnings.filterwarnings("error")
import numba as nb
from utils.logger import Logger, LogLevel


@nb.njit()
def getAvgDistInSet_nb(embeddings, mean_embedding):
    l = len(embeddings)
    embedding_mean = 0
    count = 0

    for i in nb.prange(l):
        # a, b preserved to comparability to cosine distance calculation
        # copied here because numba
        a = embeddings[i]
        b = mean_embedding
        div = (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
        if (div == 0):
            div = 1e-6
        d = np.dot(a, b)/div
        if (d > 1):
            d = 1
        if (d < -1):
            d = -1

        temp = 1/(count + 1) * (count*embedding_mean + d)
        embedding_mean = temp
        count += 1
    return embedding_mean, l

@nb.njit(parallel=True)
def intra_nb(labels, embeddings, mean_embeddings):
    dists = {}
    for i in nb.prange(len(labels)):
        label = labels[i]
        q_embeddings = embeddings[i]
        mean_embedding = mean_embeddings[i]
        (avgdist, cnt) = getAvgDistInSet_nb(q_embeddings, mean_embedding)
        dists[label] = (avgdist, cnt)
    return dists

class EmbeddingAnalyzer:

    #  ██████╗████████╗ ██████╗ ██████╗
    # ██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
    # ██║        ██║   ██║   ██║██████╔╝
    # ██║        ██║   ██║   ██║██╔══██╗
    # ╚██████╗   ██║   ╚██████╔╝██║  ██║
    #  ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝

    def __init__(self, input, dir, isNumpy, classes_path, delimiter, device, embeddingSize=512):
        # parameters
        self.device = device
        self.embeddingSize = embeddingSize
        # settings
        self.dataInitialized = False
        self.mapCount = 0
        self.printColumns = 4
        # "columns": label, instance, region, embedding, distance
        self.columns = 5
        # if textquery is found in labels, display separate histogram for match
        self.matchHistogram = True
        # histogram settings: bins min, max, num. relative / absolute mode
        self.bmin = 0
        self.bmax = 1
        self.nbins = 200
        self.relativeHistogram = True
        # display distribution fits
        self.displayFits = False
        # aggregate label
        self.aggregateName = "aggregate"
        self.aggregateLabel = -100
        # n/a label
        self.nastring = "n/a"
        self.errstring = "err"
        # store distances
        self.got_distances = False
        self.d_query = None
        self.distances = None

        self.logger = Logger("analyzer", level=LogLevel.INFO, verbosity=LogLevel.NONE)

        if(not dir and not input):
            print("No data given")
            return

        # read class files, get labels and names
        # classes is all the classes in the ocnfiguration file
        self.classData = self.parseClassFile(classes_path, delimiter)
        self.classes = self.classData[0, :].astype(dtype=np.int32)
        self.classNames = self.classData[1, :].astype(dtype=str)
        self.numClasses = max(self.classNames.shape)

        # parse the input data
        if (not dir):
            if (isNumpy):
                self.parseNumpy(input)
            else:
                self.parseCpp(input)
        else:
            files = [f for f in listdir(dir) if isfile(join(dir, f))]
            for file in files:
                if (isNumpy):
                    self.parseNumpy(dir+"/"+file, True)
                else:
                    self.parseCpp(dir+"/"+file, True)
        # select only the classes used in the data.
        # usedClasses is all of the classes present in the loaded data set
        self.usedClasses = np.unique(
            self.data[:, :, 0, 0]).astype(dtype=np.int32)
        mask = np.isin(self.usedClasses, self.classes)
        self.usedClasses = self.usedClasses[mask]
        indices = np.searchsorted(self.classes, self.usedClasses)
        self.usedClassNames = self.classNames[indices]

        # selected classes is the list of classes selected by the user
        self.selectedClassNames = np.copy(self.usedClassNames)
        self.selectedClasses = np.copy(self.usedClasses)
        self.usedClasses = np.append(self.usedClasses, self.aggregateLabel)
        self.usedClassNames = np.append(
            self.usedClassNames, self.aggregateName)


    #  █████╗ ███╗   ██╗ █████╗ ██╗  ██╗   ██╗███████╗██╗███████╗
    # ██╔══██╗████╗  ██║██╔══██╗██║  ╚██╗ ██╔╝██╔════╝██║██╔════╝
    # ███████║██╔██╗ ██║███████║██║   ╚████╔╝ ███████╗██║███████╗
    # ██╔══██║██║╚██╗██║██╔══██║██║    ╚██╔╝  ╚════██║██║╚════██║
    # ██║  ██║██║ ╚████║██║  ██║███████╗██║   ███████║██║███████║
    # ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝   ╚══════╝╚═╝╚══════╝

    def analyze(self, query, fit, distanalysis, textQuery=None, batch=False, numProcess=8, mean=False):
        # cosine distances
        distances = self.getDistances(query, batch)

        stats = {}

        # histograms
        stats = self.getStats(distances, stats, textQuery, batch, mean, query)

        # fit distributions
        if (fit):
            fits = self.getFits(distances, fit, batch)
        else:
            fits = None

        # wasserstein distances
        if (distanalysis):
            # FIXME: comparison
            map = 0
            stats = self.distributionAnalysisFromQuery(
                stats, map, textQuery, batch, numProcess)

        # print
        if (not batch):
            self.printStats(stats, fits, textQuery)

        return stats, fits

    def classification(self, queries, labels, aggregate, batch=False):
        origLabels, maxLabels = self.classifyMap(queries, labels)

        output = []
        if aggregate:
            for i in tqdm(range(len(labels)), leave=False):
                label = labels[i]
                idx = origLabels[:, :] == label
                gt = origLabels[idx]
                pred = maxLabels[idx]
                outitem = self.classifyItem(gt, pred, "all", label, batch)
                if outitem:
                    output.append(outitem)
        else:
            for m in tqdm(range(self.data.shape[0]), leave=False):
                for i in tqdm(range(len(labels)), leave=False):
                    label = labels[i]
                    idx = origLabels[m, :] == label
                    gt = origLabels[m,idx]
                    pred = maxLabels[m,idx]
                    outitem = self.classifyItem(gt, pred, m, label, batch)
                    if outitem:
                        output.append(outitem)
        return output

    def classifyItem(self, gt, pred, m, label, batch):
        if(gt.size == 0):
            return None
        accuracy, macro_averaged_precision, micro_averaged_precision, macro_averaged_recall, micro_averaged_recall, macro_averaged_f1, micro_averaged_f1, iou = self.classify(gt, pred)
        outitem = []
        outitem.append(m)
        outitem.append(label)
        cls = self.getNameForLabel(label)
        outitem.append(cls)
        outitem.append(accuracy)
        outitem.append(macro_averaged_precision)
        outitem.append(micro_averaged_precision)
        outitem.append(macro_averaged_recall)
        outitem.append(micro_averaged_recall)
        outitem.append(macro_averaged_f1)
        outitem.append(micro_averaged_f1)
        outitem.append(iou)

        if(not batch):
            print("Label:", cls)
            print("accuracy", accuracy)
            print("macro_averaged_precision", macro_averaged_precision)
            print("micro_averaged_precision", micro_averaged_precision)
            print("macro_averaged_recall", macro_averaged_recall)
            print("micro_averaged_recall", micro_averaged_recall)
            print("macro_averaged_f1", macro_averaged_f1)
            print("micro_averaged_f1", micro_averaged_f1)
            print("IoU", iou)

        return outitem

    def classifyMap(self, queries, labels):
        # get the predicted embeddings
        embeddings = self.data[:, :, 3, :]
        orig_shape = np.copy((embeddings.shape[0], embeddings.shape[1]))
        embeddings = embeddings.reshape(-1, self.embeddingSize)
        distances = np.full(
            (len(queries), embeddings.shape[0]), -1, dtype=np.float32)

        for i in tqdm(range(len(queries)), desc="Classification", leave=False):
            query = queries[i]
            for j in tqdm(range(embeddings.shape[0]), disable=True, desc="Distances", leave=False):
                e = embeddings[j, :]
                d = self.distance(e, query)
                distances[i,j] = d

        # find the argmax of the embeddings (label with highest value)
        maxes = np.argmax(distances, axis=0)
        maxLabels = labels[maxes]
        maxLabels = maxLabels.reshape(orig_shape)
        origLabels = self.data[:,:,0,0]

        return origLabels, maxLabels

    def confusion(self, queries, labels):
        origLabels = np.load(os.environ['VLMAPS_DIR'] + "/data/tmp/origLabels.npy", allow_pickle=True)
        maxLabels = np.load(os.environ['VLMAPS_DIR'] + "/data/tmp/maxLabels.npy", allow_pickle=True)

        matrix = metrics.confusion_matrix(origLabels, maxLabels, labels=labels, normalize="true")
        return matrix, labels

    def instance_statistics(self, textQuery, batch=False):
        stats = {}
        label = -1
        idx = 0
        found = False
        for idx in range(len(self.usedClasses)):
            name = self.usedClassNames[idx]
            if (name == textQuery):
                q_label = self.usedClasses[idx]
                found = True
                break

        if (not found):
            if (not batch):
                print("no query label found")
            return stats

        print("have", self.mapCount, "maps")

        for map in tqdm(range(self.mapCount), disable=batch, leave=False, desc="distribution analysis: map"):
            for q_label in tqdm(self.selectedClasses, disable=batch, leave=False, desc="distribution analysis: q label"):
                q_instances = self.getInstances(q_label, map)
                for q_instance in tqdm(q_instances, disable=batch, leave=False, desc="distribution analysis: q instance"):
                    stats[map,q_label,q_instance] = {}

                    if (int(label) == int(self.aggregateLabel)):
                        continue

                    q_embeddings = self.getEmbeddings(q_label, map, q_instance, False)
                    if (q_embeddings.size == 0):
                        continue

                    for label in tqdm(self.selectedClasses, disable=batch, leave=False, desc="distribution analysis: label"):
                        instances = self.getInstances(label, map)
                        for instance in tqdm(instances, disable=batch, leave=False, desc="distribution analysis: instance"):

                            embeddings = self.getEmbeddings(label, map, instance, True)
                            if (embeddings.size == 0):
                                continue

                            verbose = False
                            stats[map,q_label,q_instance][label, instance] = {}

                            stats[map, q_label, q_instance][label, instance]["p_wsd"] = self.getMetric(
                                self.parametric_wsd, embeddings, q_embeddings, verbose)


        return stats, self.mapCount, self.selectedClasses

    #DEBUG
    def printInstanceStatistics(self):
        tot = 0
        for map in range(self.mapCount):
            for label in self.selectedClasses:
                name = self.getNameForLabel(label)
                instances = self.getInstances(label, map)
                print("="*10)
                print("Map", map, "label", label, name, ":",
                      len(instances), "instances")
                tot += len(instances)
        print("Total", tot, "instances")

    # ████████╗███████╗███████╗████████╗███████╗
    # ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝
    #    ██║   █████╗  ███████╗   ██║   ███████╗
    #    ██║   ██╔══╝  ╚════██║   ██║   ╚════██║
    #    ██║   ███████╗███████║   ██║   ███████║
    #    ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝


    # https://en.wikipedia.org/wiki/Nonparametric_statistics
    # Mann-Whitney: Mean is different
    # Tukey-Duckworth: Mean is different
    # Welch: Mean is different
    # Kolmogorov-Smirnov: do samples originate from the same distribution
    # Kruskal-Wallis: do samples originate from the same distribution
    def statistical_tests(self, query, textQuery, map, batch=False):
        wx = self.wilcoxon(query, textQuery, map, batch)
        we = self.welch(query, textQuery, map, batch)
        ks = self.kolmogorov(query, textQuery, map, batch)
        kw = self.kruskal(query, textQuery, map, batch)
        mw = self.whitney(query, textQuery, map, batch)
        td = self.tukey(query, textQuery, map, batch)
        return wx, we, ks, kw, mw, td


    # https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test
    def whitney(self, query, textQuery, map, batch=False):
        return self.statTest(self.impl_whitney, "Mann-Whitney U test", query, textQuery, map, batch)

    # https://en.wikipedia.org/wiki/Tukey%E2%80%93Duckworth_test
    def tukey(self, query, textQuery, map, batch=False):
        return self.statTest(self.impl_tukey, "Tukey-Duckworth test", query, textQuery, map, batch)

    # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    def kolmogorov(self, query, textQuery, map, batch=False):
        return self.statTest(self.impl_kolmogorov, "Kolmogorov-Smirnov test", query, textQuery, map, batch)

    # https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_one-way_analysis_of_variance
    def kruskal(self, query, textQuery, map, batch=False):
        return self.statTest(self.impl_kruskal, "Kruskal-Wallis test", query, textQuery, map, batch)

    # https://en.wikipedia.org/wiki/Welch%27s_t-test
    def welch(self, query, textQuery, map, batch=False):
        return self.statTest(self.impl_welch, "Welch's t-test", query, textQuery, map, batch)

    def statTest(self, f, title, query, textQuery, map, batch=False):
        matched, others = self.getMatchesAndOthers(
            query, textQuery, map, batch)
        match_size = matched.size
        others_size = others.size
        res = f(matched, others)
        statistic = res.statistic
        pvalue = res.pvalue

        # for tukey test
        if (isinstance(statistic, np.ndarray)):
            statistic = statistic[0, 1]
        if (isinstance(pvalue, np.ndarray)):
            pvalue = pvalue[0, 1]

        if (not batch):
            print("*"*40)
            print(title)
            print("statistic:", statistic)
            print("pvalue:   ", pvalue)
        return (statistic, pvalue, match_size, others_size)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    def impl_whitney(self, match, others):
        return scipy.stats.mannwhitneyu(match, others, alternative="greater")

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html
    def impl_tukey(self, match, others):
        return scipy.stats.tukey_hsd(match, others)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
    def impl_kolmogorov(self, match, others):
        #note: this forks other way round, less means match >= others! Soo documentation.
        return scipy.stats.ks_2samp(match, others, alternative="less")
        #return scipy.stats.kstest(match, others, alternative="less")

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
    def impl_kruskal(self, match, others):
        return scipy.stats.kruskal(match, others)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    def impl_welch(self, match, others):
        return scipy.stats.ttest_ind(match, others, alternative="greater")

    def impl_wilcoxon(self, match, others, densityHistograms=True):
        match_histogram, other_histogram, match_edges, other_edges = self.bin(
            match, others, densityHistograms)

        # FIXME this should not be here! this is just for outlier rejection, the root cause must be solved
        idx = match_edges > 0.5
        match_histogram = match_histogram[idx[:-1]]
        other_histogram = other_histogram[idx[:-1]]
        match_edges = match_edges[idx]
        other_edges = other_edges[idx]

        return scipy.stats.wilcoxon(match_histogram, other_histogram, alternative="two-sided")

    def wilcoxon(self, query, textQuery, map, batch=False):
        if (textQuery not in self.usedClassNames):
            print("Unknown query")
            return None, None, None, None

        # get matches and distances
        matched, others = self.getMatchesAndOthers(
            query, textQuery, map, batch)
        densityHistograms = True

        match_size = matched.size
        others_size = others.size

        match_histogram, other_histogram, match_edges, other_edges = self.bin(
            matched, others, densityHistograms)

        # FIXME this should not be here! this is just for outlier rejection, the root cause must be solved
        idx = match_edges > 0.5
        match_histogram = match_histogram[idx[:-1]]
        other_histogram = other_histogram[idx[:-1]]
        match_edges = match_edges[idx]
        other_edges = other_edges[idx]

        res = scipy.stats.wilcoxon(match_histogram, other_histogram)
        statistic = res.statistic
        pvalue = res.pvalue

        if not batch:
            print("*"*40)
            print("Wilcoxon")
            print("statistic:", statistic)
            print("pvalue:   ", pvalue)
            mapstr = "map " + str(map) if map != -1 else "all maps"
            ylabel = "density" if densityHistograms else "count"
            xlabel = "cosine distance"

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.suptitle("Wilcoxon test " + mapstr)
            # match
            ax1.set_title("matches: "+textQuery)
            ax1.bar(match_edges[:-1], match_histogram, width=0.01, color='r')
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            # other
            ax2.set_title("non matching")
            ax2.bar(other_edges[:-1], other_histogram, width=0.01, color='b')
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel(ylabel)
            # combo
            ax3.set_title("combo")
            ax3.step(match_edges[:-1], match_histogram, 'r')
            ax3.step(other_edges[:-1], other_histogram, 'b')

            # orientation='horizontal', hatch='//'
            ax3.set_xlabel(xlabel)
            ax3.set_ylabel(ylabel)
            plt.show()
        return (statistic, pvalue, match_size, others_size)

    #  ██████╗██████╗  █████╗
    # ██╔════╝██╔══██╗██╔══██╗
    # ██║     ██║  ██║███████║
    # ██║     ██║  ██║██╔══██║
    # ╚██████╗██████╔╝██║  ██║
    #  ╚═════╝╚═════╝ ╚═╝  ╚═╝


    def mp_crossDistributionAnalysis(self, batch=False, numProcess=16):
        return self.cda(self.mp_distributionAnalysis, batch, numProcess)

    def crossDistributionAnalysis(self, batch=False):
        return self.cda(self.distributionAnalysis, batch, None)

    def cda(self, f, batch, numProcess):
        allStats = {}
        for map in tqdm(range(self.mapCount), disable=True, leave=False, desc="cross-distribution analysis: map"):
            partialStats = self.cda_single(map, f, batch, numProcess)
            for (m, l) in partialStats:
                allStats[m, l] = partialStats[m, l]
        return allStats

    def mp_crossDistributionAnalysisSingleMap(self, map, batch=False, numProcess=16):
        return self.cda_single(map, self.mp_distributionAnalysis, batch, numProcess)

    def crossDistributionAnalysisSingleMap(self, map, batch=False):
        return self.cda_single(map, self.distributionAnalysis, batch, None)

    def cda_single(self, map, f, batch, numProcess):
        allStats = {}
        for label in tqdm(self.selectedClasses, disable=False, leave=False, desc="cross-distribution analysis: label"):
            q_embeddings = self.getEmbeddings(label, map)
            stats = self.initStats()
            stats = f(stats, q_embeddings, batch, numProcess)
            allStats[map, label] = stats
        return allStats

    def distributionAnalysisFromQuery(self, stats, map, textQuery, batch=False, numProcess=8):
        q_label = -1
        idx = 0
        found = False
        for idx in range(len(self.usedClasses)):
            name = self.usedClassNames[idx]
            if (name == textQuery):
                q_label = self.usedClasses[idx]
                found = True
                break

        if (not found):
            if (not batch):
                print("no query label found")
            return stats

        q_embeddings = self.getEmbeddings(q_label, map)
        return self.mp_distributionAnalysis(stats, q_embeddings, batch, numProcess)

    def mp_distributionAnalysis_process(self, map, label, q_embeddings):
        if (int(label) == int(self.aggregateLabel)):
            return None
        embeddings = self.getEmbeddings(label, map)
        if (embeddings.size == 0):
            return None
        verbose = False
        stats_ = {}
        stats_["b_wsd"] = self.getMetric(
            self.binned_wsd, embeddings, q_embeddings, verbose)
        stats_["b_kld"] = self.getMetric(
            self.binned_kld, embeddings, q_embeddings, verbose)
        stats_["b_jsd"] = self.getMetric(
            self.binned_jsd, embeddings, q_embeddings, verbose)
        stats_["b_bha"] = self.getMetric(
            self.binned_bha, embeddings, q_embeddings, verbose)
        stats_["p_wsd"] = self.getMetric(
            self.parametric_wsd, embeddings, q_embeddings, verbose)
        stats_["p_kld"] = self.getMetric(
            self.parametric_kld, embeddings, q_embeddings, verbose)
        stats_["p_jsd"] = self.getMetric(
            self.parametric_jsd, embeddings, q_embeddings, verbose)
        return (map, label, stats_)

    def mp_distributionAnalysis(self, stats, q_embeddings, batch=False, numProcess=8):
        allStats = Parallel(n_jobs=numProcess)(delayed(self.mp_distributionAnalysis_process)(map, label, q_embeddings)
                                               for map in tqdm(range(self.mapCount), disable=batch, leave=False) for label in tqdm(self.selectedClasses, disable=batch, leave=False))
        for var in allStats:
            if (not var):
                continue
            (map, label, stat) = var
            stats[map, label]["b_wsd"] = stat["b_wsd"]
            stats[map, label]["b_kld"] = stat["b_kld"]
            stats[map, label]["b_jsd"] = stat["b_jsd"]
            stats[map, label]["b_bha"] = stat["b_bha"]
            stats[map, label]["p_wsd"] = stat["p_wsd"]
            stats[map, label]["p_kld"] = stat["p_kld"]
            stats[map, label]["p_jsd"] = stat["p_jsd"]

        return stats

    def getMetric(self, f, embeddings, q_embeddings, verbose=False):
        if (embeddings.size <= 0 or q_embeddings.size <= 0):
            return self.nastring

        try:
            val = f(embeddings, q_embeddings)
        except Exception as e:
            errorstring = str(e) + " in " + f.__name__
            self.logger.warning(errorstring)
            val = self.errstring

        if (verbose):
            print(f.__name__, val, type(val))

        if(str(val) != self.errstring and not isinstance(val, np.float64)):
            errorstring = str(val) + " is type " + str(type(val)) + " in " + f.__name__
            self.logger.warning(errorstring)
            val = self.errstring

        return val

    def distributionAnalysis(self, stats_, q_embeddings, batch=False, numProcess=None):
        for map in tqdm(range(self.mapCount), disable=True, leave=False, desc="distribution analysis: map"):
            for label in tqdm(self.selectedClasses, disable=True, leave=False, desc="distribution analysis: label"):
                if (int(label) == int(self.aggregateLabel)):
                    continue
                embeddings = self.getEmbeddings(label, map)
                if (embeddings.size == 0):
                    continue
                verbose = False
                stats_[map, label]["b_wsd"] = self.getMetric(
                    self.binned_wsd, embeddings, q_embeddings, verbose)
                stats_[map, label]["b_kld"] = self.getMetric(
                    self.binned_kld, embeddings, q_embeddings, verbose)
                stats_[map, label]["b_jsd"] = self.getMetric(
                    self.binned_jsd, embeddings, q_embeddings, verbose)
                stats_[map, label]["b_bha"] = self.getMetric(
                    self.binned_bha, embeddings, q_embeddings, verbose)
                stats_[map, label]["p_wsd"] = self.getMetric(
                    self.parametric_wsd, embeddings, q_embeddings, verbose)
                stats_[map, label]["p_kld"] = self.getMetric(
                    self.parametric_kld, embeddings, q_embeddings, verbose)
                stats_[map, label]["p_jsd"] = self.getMetric(
                    self.parametric_jsd, embeddings, q_embeddings, verbose)
        return stats_

    # ██╗███╗   ██╗████████╗██████╗  █████╗ ███╗   ███╗ █████╗ ██████╗
    # ██║████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗████╗ ████║██╔══██╗██╔══██╗
    # ██║██╔██╗ ██║   ██║   ██████╔╝███████║██╔████╔██║███████║██████╔╝
    # ██║██║╚██╗██║   ██║   ██╔══██╗██╔══██║██║╚██╔╝██║██╔══██║██╔═══╝
    # ██║██║ ╚████║   ██║   ██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║██║
    # ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝

    def intraMapAnalysis(self, map, batch=False):
        return self.intraMapAnalysis_slow(map)

    def intraMapAnalysis_fast(self, map):
        dists = {}

        all_embeddings = self.getAllEmbeddings(map)
        all_mean = self.getMean(all_embeddings)
        (avgdist_map, count) = getAvgDistInSet_nb(all_embeddings, all_mean)
        dists[self.aggregateLabel] = (avgdist_map, count)

        embeddings = []
        labels = []
        mean_embeddings = []
        for label in self.selectedClasses:
            em = self.getEmbeddings(label, map)
            embeddings.append(em)
            labels.append(label)
            mean_embeddings.append(self.getMean(em))

        ldists = intra_nb(labels, embeddings, mean_embeddings)

        for key in ldists:
            dists[key] = ldists[key]
        return dists

    def intraMapAnalysis_slow(self, map):
        dists = {}

        all_embeddings = self.getAllEmbeddings(map)
        all_mean = self.getMean(all_embeddings)
        (avgdist_map, count) = self.getAvgDistInSet(all_embeddings, all_mean)
        dists[self.aggregateLabel] = (avgdist_map, count)

        embeddings = []
        labels = []
        mean_embeddings = []
        for label in self.selectedClasses:
            em = self.getEmbeddings(label, map)
            embeddings.append(em)
            labels.append(label)
            mean_embeddings.append(self.getMean(em))

        ldists = self.intra(labels, embeddings, mean_embeddings)

        for key in ldists:
            dists[key] = ldists[key]
        return dists

    def intra(self, labels, embeddings, mean_embeddings):
        dists = {}
        for i in tqdm(range(len(labels))):
            label = labels[i]
            q_embeddings = embeddings[i]
            mean_embedding = mean_embeddings[i]
            (avgdist, cnt) = self.getAvgDistInSet(q_embeddings, mean_embedding)
            dists[label] = (avgdist, cnt)
        return dists

    def getAvgDistInSet(self, embeddings, mean_embedding):
        l = len(embeddings)
        embedding_mean = 0
        count = 0

        for i in tqdm(range(l)):
            d = self.distance(embeddings[i], mean_embedding)
            temp = 1/(count + 1) * (count*embedding_mean + d)
            embedding_mean = temp
            count += 1
        return embedding_mean, l


    # ███████╗████████╗ █████╗ ████████╗███████╗
    # ██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝
    # ███████╗   ██║   ███████║   ██║   ███████╗
    # ╚════██║   ██║   ██╔══██║   ██║   ╚════██║
    # ███████║   ██║   ██║  ██║   ██║   ███████║
    # ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝

    def getStats(self, distances, stats, textQuery, batch, mean, query):
        gotMatch = False
        legend = []
        if not batch:
            if (textQuery is not None and self.matchHistogram):
                fig, (ax1, ax2) = plt.subplots(1, 2)
            else:
                fig, ax1 = plt.subplots()

        for map in tqdm(range(self.mapCount), leave=False, disable=batch, desc="stats: map"):
            idx = 0
            for label in tqdm(self.selectedClasses, leave=False, disable=batch, desc="stats: label"):
                name = self.selectedClassNames[idx]
                if (not mean):
                    dist = self.getDistParam(distances, label, map)
                else:
                    embs = self.getEmbeddings(label, map)
                    mn = self.getMean(embs)
                    dist = self.distance(mn, query)
                match = textQuery is not None and textQuery == name
                if (not batch):
                    self.plot(ax1, dist, match)
                    if (match and self.matchHistogram):
                        self.plot(ax2, dist, match and self.mapCount == 1)
                        gotMatch = True
                obj = self.stats(dist, name)
                stats[map, label] = obj
                legend.append(str(map) + "-" + name)
                idx += 1

        if (not batch):
            ax1.legend(legend, loc="upper left")
            if (gotMatch and self.matchHistogram):
                ax2.legend([textQuery], loc="upper left")
            plt.show()

        return stats

    def stats(self, dist, name):
        obj = self.getEmptyStatObject()

        if (dist.size > 0):
            dist = dist[np.invert(np.isnan(dist))]
            dist = dist[np.invert(np.isinf(dist))]
            std = np.std(dist)
            mean = np.mean(dist)
            mx = np.max(dist)
            mn = np.min(dist)
            obj["name"] = name
            obj["mean"] = mean
            obj["std"] = std
            obj["min"] = mn
            obj["max"] = mx

        return obj

    def getEmptyStatObject(self):
        obj = {"name": self.nastring, "mean": self.nastring, "std": self.nastring, "max": self.nastring, "min": self.nastring, "b_kld": self.nastring, "b_jsd": self.nastring, "b_bha": self.nastring, "b_wsd": self.nastring,
               "p_kld": self.nastring, "p_jsd": self.nastring, "p_bha": self.nastring, "p_wsd": self.nastring}
        return obj

    def initStats(self):
        stats_ = {}
        for map in range(self.mapCount):
            idx = 0
            for label in self.selectedClasses:
                obj = self.getEmptyStatObject()
                name = self.selectedClassNames[idx]
                obj["name"] = name
                stats_[map, label] = obj
                idx += 1
        return stats_

    #  ██████╗██╗      █████╗ ███████╗███████╗██╗███████╗██╗ ██████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
    # ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔════╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
    # ██║     ██║     ███████║███████╗███████╗██║█████╗  ██║██║     ███████║   ██║   ██║██║   ██║██╔██╗ ██║
    # ██║     ██║     ██╔══██║╚════██║╚════██║██║██╔══╝  ██║██║     ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
    # ╚██████╗███████╗██║  ██║███████║███████║██║██║     ██║╚██████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
    #  ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝     ╚═╝ ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝


    def classify(self, y_true, y_pred):
        # correct predicions / all predictions
        # TP = true positive
        # TN = true negatice
        # FP = false positive
        # FN = false negative

        # Accuracy Score = (TP + TN) / (TP + TN + FP + FN)
        accuracy = metrics.accuracy_score(y_true, y_pred)

        # Precision = TP / (TP + FP)
        # Macro averaged precision: calculate precision for all classes individually and then average them
        macro_averaged_precision = metrics.precision_score(
            y_true, y_pred, average='macro')

        # Micro averaged precision: calculate class wise true positive and false positive and then use that to calculate overall precision
        micro_averaged_precision = metrics.precision_score(
            y_true, y_pred, average='micro')

        # Recall = TP / (TP + FN)
        macro_averaged_recall = metrics.recall_score(
            y_true, y_pred, average='macro')
        micro_averaged_recall = metrics.recall_score(
            y_true, y_pred, average='micro')

        # F1 = 2PR / (P + R)
        macro_averaged_f1 = metrics.f1_score(y_true, y_pred, average='macro')
        micro_averaged_f1 = metrics.f1_score(y_true, y_pred, average='micro')

        # Intersect over Union (Jaccard index)
        intersection = (y_true == y_pred).sum()
        union = y_true.size + y_pred.size - intersection
        iou = intersection/union

        return accuracy, macro_averaged_precision, micro_averaged_precision, macro_averaged_recall, micro_averaged_recall, macro_averaged_f1, micro_averaged_f1, iou

    # ███████╗██╗████████╗███████╗
    # ██╔════╝██║╚══██╔══╝██╔════╝
    # █████╗  ██║   ██║   ███████╗
    # ██╔══╝  ██║   ██║   ╚════██║
    # ██║     ██║   ██║   ███████║
    # ╚═╝     ╚═╝   ╚═╝   ╚══════╝

    def getFits(self, distances, fit, batch):
        fits = {}
        for map in tqdm(range(self.mapCount), leave=False, disable=batch, desc="fits: map"):
            idx = 0
            for label in tqdm(self.selectedClasses, leave=False, disable=batch, desc="fits: label"):
                name = self.selectedClassNames[idx]
                dist = self.getDistParam(distances, label, map)
                fitObj, success = self.fit(dist, label, name, batch)
                fits[map, label] = fitObj
                if (self.displayFits and success and not batch):
                    plt.title(name)
                    plt.show()
                idx += 1

        return fits

    def fit(self, dist, label, name, batch=False):
        success = True
        if (batch):
            self.disablePrint()
        try:
            f = Fitter(dist, distributions=get_common_distributions())
            f.fit()
            if (not batch):
                f.summary()
            best = f.get_best(method='sumsquare_error')
            obj = {"name": name, "best": best}
        except:
            obj = {"name": name, "best": None}
            success = False
        if (batch):
            self.enablePrint()
        return obj, success

    # ███╗   ███╗███████╗ █████╗ ███████╗██╗   ██╗██████╗ ███████╗███████╗
    # ████╗ ████║██╔════╝██╔══██╗██╔════╝██║   ██║██╔══██╗██╔════╝██╔════╝
    # ██╔████╔██║█████╗  ███████║███████╗██║   ██║██████╔╝█████╗  ███████╗
    # ██║╚██╔╝██║██╔══╝  ██╔══██║╚════██║██║   ██║██╔══██╗██╔══╝  ╚════██║
    # ██║ ╚═╝ ██║███████╗██║  ██║███████║╚██████╔╝██║  ██║███████╗███████║
    # ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝

    def distPrint(self, p, title):
        mn = np.min(p)
        mx = np.max(p)
        s = np.sum(p)
        print(title, "shape", p.shape, "min", mn, "max", mx, "sum", s)

    # cosine distance
    def distance(self, a, b):
        div = (np.linalg.norm(a, ord=2) * np.linalg.norm(b, ord=2))
        if (div == 0):
            div = 1e-6
        d = np.dot(a, b)/div
        if (d > 1):
            d = 1
        if (d < -1):
            d = -1
        return d

    # ██████╗ ██╗███╗   ██╗███╗   ██╗███████╗██████╗
    # ██╔══██╗██║████╗  ██║████╗  ██║██╔════╝██╔══██╗
    # ██████╔╝██║██╔██╗ ██║██╔██╗ ██║█████╗  ██║  ██║
    # ██╔══██╗██║██║╚██╗██║██║╚██╗██║██╔══╝  ██║  ██║
    # ██████╔╝██║██║ ╚████║██║ ╚████║███████╗██████╔╝
    # ╚═════╝ ╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝╚═════╝

    def bin(self, p, q, density=True):
        pmn = np.min(p)
        qmn = np.min(q)
        pmx = np.max(p)
        qmx = np.max(q)
        mn = pmn if pmn < qmn else qmn
        mx = pmx if pmx > qmx else qmx

        # squre root
        n = np.size(p) if np.size(p) > np.size(q) else np.size(q)
        k = np.sqrt(n)
        k = int(np.ceil(k))

        # sturges
        # k = np.log(n)
        # k = int(np.ceil(k)) + 1

        # Freedman-Diaconis
        # q75, q25 = np.percentile(p, [75 ,25])
        # iqr = q75 - q25
        # k = 2*iqr/(n**(1./3.))
        # k = int(np.ceil(k))
        (hp, pedges) = np.histogram(p, bins=k, range=(mn, mx), density=density)
        (hq, qedges) = np.histogram(q, bins=k, range=(mn, mx), density=density)

        return hp, hq, pedges, qedges

    def binned_run(self, p, q, f):
        #2D
        if(len(p.shape) > 1):
            assert (p.shape[1] == q.shape[1])
            l = p.shape[1]
            vals = np.zeros((l))
            for i in range(l):
                hp, hq, _, _ = self.bin(p[:, i], q[:, i])
                vals[i] = f(hp, hq)
        #1D
        else:
            hp, hq, _, _ = self.bin(p, q)
            vals = f(hp, hq)
        return vals

    def aggregate_dimensions(self, vals):
        return np.sum(vals)

    def binned_kld(self, p, q):
        vals = self.binned_run(p, q, self.kullback_leibler)
        if(isinstance(vals, float)):
            return vals
        else:
            return self.aggregate_dimensions(vals)

    def binned_jsd(self, p, q):
        vals = self.binned_run(p, q, self.jensen_shannon)
        if(isinstance(vals, float)):
            return vals
        else:
            return self.aggregate_dimensions(vals)

    def binned_bha(self, p, q):
        vals = self.binned_run(p, q, self.bhattacharyya)
        if(isinstance(vals, float)):
            return vals
        else:
            return self.aggregate_dimensions(vals)

    def binned_wsd(self, p, q):
        vals = self.binned_run(p, q, self.wasserstein)
        if(isinstance(vals, float)):
            return vals
        else:
            return self.aggregate_dimensions(vals)

    def kullback_leibler(self, p, q):
        if (p.shape != q.shape):
            return -1
        p = np.where(p == 0, p + 1e-3, p)
        q = np.where(q == 0, q + 1e-3, q)
        ratio = p/q
        if(np.any(ratio < 0)):
            return float('nan')
        return np.sum(p*np.log(ratio))

    def jensen_shannon(self, p, q):
        m = 0.5*(p+q)
        return 0.5 * (self.kullback_leibler(p, m) + self.kullback_leibler(q, m))

    def bhattacharyya(self, p, q):
        if (p.shape != q.shape):
            return -1
        s = np.sum(np.sqrt(p * q))
        if(np.any(s < 0)):
            return float('nan')
        return -np.log(s)

    def wasserstein(self, p, q):
        if (p.shape != q.shape):
            return -1
        return np.mean(np.abs((np.sort(p)-np.sort(q))))

    def fit_normal(self, p):
        mean = np.mean(p, axis=0)
        cov = np.cov(p, rowvar=False)
        return mean, cov

    # ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗
    # ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝
    # ██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   ██████╔╝██║██║
    # ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║
    # ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗
    # ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝

    def printmat(self, M, name):
        print(name + " rank", np.linalg.matrix_rank(M), "shape", M.shape)

    def getPrincipalSquareRoot(self, M):
        (evals, evecs) = np.linalg.eig(M)
        V = np.stack(evecs, axis=0)
        D = np.diag(evals)
        S = np.sqrt(D)
        R = V @ S @ np.linalg.inv(V)
        return R

    def parametric_wsd(self, p, q):
        (p_mean, p_cov) = self.fit_normal(p)
        (q_mean, q_cov) = self.fit_normal(q)

        # https://en.wikipedia.org/wiki/Wasserstein_metric
        if(p_cov.size == 1):
            if(q_cov < 0):
                return float('nan')
            sqQ = np.sqrt(q_cov)
            C = sqQ * p_cov * sqQ
            if (C < 0):
                return float('nan')
            sqC = np.sqrt(C)

            val = np.linalg.norm(p_mean - q_mean) + p_cov + q_cov - 2*sqC

        else:
            sqQ = self.getPrincipalSquareRoot(q_cov)
            C = sqQ @ p_cov @ sqQ
            sqC = self.getPrincipalSquareRoot(C)
            val = np.linalg.norm(p_mean - q_mean, ord=2) + \
                np.trace(p_cov + q_cov - 2*sqC)
        return np.real(val)

    def parametric_kld(self, p, q):
        (p_mean, p_cov) = self.fit_normal(p)
        (q_mean, q_cov) = self.fit_normal(q)
        n = self.embeddingSize
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions

        # p = 0, q = 1
        if(p_cov.size == 1):
            ratio = q_cov/p_cov
            if(ratio < 0):
                return float('nan')
            return np.log(ratio) + (p_cov**2 + (p_mean - q_mean)**2)/(2*q_cov**2) - 0.5
        else:
            # try:
            L0 = np.linalg.cholesky(p_cov)
            L1 = np.linalg.cholesky(q_cov)
            M = np.linalg.solve(L1, L0)
            y = np.linalg.solve(L1, (q_mean - p_mean))
            ratio = np.diagonal(L1)/np.diagonal(L0)
            if(np.any(ratio < 0)):
                return float('nan')
            val = 0.5 * (np.sum(np.diagonal(M)**2) - n + np.linalg.norm(y) ** 2 + 2 * np.sum(np.log(ratio)))
            return val

    def parametric_jsd(self, p, q):
        # FIXME: this is not be the right way
        m = 0.5*np.concatenate((p, q), axis=0)
        return 0.5 * (self.parametric_kld(p, m) + self.parametric_kld(q, m))

    # ██████╗ ██╗      ██████╗ ████████╗
    # ██╔══██╗██║     ██╔═══██╗╚══██╔══╝
    # ██████╔╝██║     ██║   ██║   ██║
    # ██╔═══╝ ██║     ██║   ██║   ██║
    # ██║     ███████╗╚██████╔╝   ██║
    # ╚═╝     ╚══════╝ ╚═════╝    ╚═╝

    def plot(self, ax, dist, match):
        htype = "step"
        if (match):
            htype = "stepfilled"
        bins = np.linspace(self.bmin, self.bmax, self.nbins)
        try:
            ax.hist(dist, bins=bins, histtype=htype,
                    density=self.relativeHistogram, linewidth=2)
        except RuntimeWarning:
            pass

    def printRow(self, label, max, hdr, val, end):
        # header
        if (label in max):
            print(f'{"* " + hdr + ": ":>15}', end="")
        else:
            print(f'{hdr+": ":>15}', end="")

        # value
        if (val == self.nastring or val == self.errstring):
            print(f'{val:<15}', end=end)
        else:
            if (isinstance(val, complex)):
                print(f'{"{:.3f}".format(val):<15}', end=end)
            else:
                print(f'{"{:.6f}".format(val):<15}', end=end)

    def getExtremum(self, field, type, stat, label):
        val = stat[field]
        if not field in self.initialized:
            self.initialized[field] = True
            self.extremum[field] = -1e6 if type == "max" else 1e6
            self.extremumLabels[field] = []

        if (val == self.nastring or val == self.errstring or np.isnan(val) or np.isinf(val)):
            return self.extremumLabels[field]

        if (type == "max"):
            add = self.extremum[field] <= val
            update = self.extremum[field] < val
        else:
            add = self.extremum[field] >= val
            update = self.extremum[field] > val
        if (update):
            self.extremum[field] = val
            self.extremumLabels[field].clear()
            self.extremumLabels[field] = [label]
        elif (add):
            self.extremumLabels[field].append(label)

        return self.extremumLabels[field]

    def printStats(self, stats, fits, textQuery=None):
        cols = self.printColumns - 1

        rows = 11
        if (fits is not None):
            rows += 4

        labels = []
        self.initialized = {}
        self.extremum = {}
        self.extremumLabels = {}
        # FIXME: proper validity check
        for map, label in stats:
            name = stats[map, label]["name"]
            if (name == self.nastring):
                continue

            labels.append((map, label))
            max_mean_label = self.getExtremum(
                "mean", "max", stats[map, label], label)
            max_std_label = self.getExtremum(
                "std", "min", stats[map, label], label)
            max_max_label = self.getExtremum(
                "max", "max", stats[map, label], label)
            max_min_label = self.getExtremum(
                "min", "min", stats[map, label], label)
            if (textQuery != stats[map, label]["name"]):
                # binned
                max_b_wsd_label = self.getExtremum(
                    "b_wsd", "min", stats[map, label], label)
                max_b_kld_label = self.getExtremum(
                    "b_kld", "min", stats[map, label], label)
                max_b_jsd_label = self.getExtremum(
                    "b_jsd", "min", stats[map, label], label)
                max_b_bha_label = self.getExtremum(
                    "b_bha", "min", stats[map, label], label)
                # parametric
                max_p_wsd_label = self.getExtremum(
                    "p_wsd", "min", stats[map, label], label)
                max_p_kld_label = self.getExtremum(
                    "p_kld", "min", stats[map, label], label)
                max_p_jsd_label = self.getExtremum(
                    "p_jsd", "min", stats[map, label], label)

        col = 0
        loopdone = False
        for idx in range(len(labels)):
            for r in range(rows+1):
                for c in range(cols+1):
                    labelIdx = idx * (cols+1) + c
                    if (labelIdx >= len(labels)):
                        loopdone = True
                        continue

                    map, label = labels[labelIdx]
                    stat = stats[map, label]
                    if (col < cols and labelIdx + 1 < len(labels)):
                        end = ""
                        col += 1
                    else:
                        end = "\n"
                        col = 0

                    if (r == 0):
                        print(f'{"Map: ":>15}', end="")
                        print(f'{map:<15}', end=end)
                    elif (r == 1):
                        if (textQuery is not None and textQuery == stat["name"]):
                            print(f'{"--> Class: ":>15}', end="")
                            print(f'{stat["name"]:<15}', end=end)
                        else:
                            print(f'{"Class: ":>15}', end="")
                            print(f'{stat["name"]:<15}', end=end)
                    elif (r == 2):
                        self.printRow(label, max_mean_label,
                                      "Mean", stat["mean"], end)
                    elif (r == 3):
                        self.printRow(label, max_std_label,
                                      "Std", stat["std"], end)
                    elif (r == 4):
                        self.printRow(label, max_min_label,
                                      "Min", stat["min"], end)
                    elif (r == 5):
                        self.printRow(label, max_max_label,
                                      "Max", stat["max"], end)
                    elif (r == 6):
                        self.printRow(label, max_b_wsd_label,
                                      "B WSD", stat["b_wsd"], end)
                    elif (r == 7):
                        self.printRow(label, max_b_kld_label,
                                      "B KLD", stat["b_kld"], end)
                    elif (r == 8):
                        self.printRow(label, max_b_jsd_label,
                                      "B JSD", stat["b_jsd"], end)
                    elif (r == 9):
                        self.printRow(label, max_b_bha_label,
                                      "B BHA", stat["b_bha"], end)
                    elif (r == 10):
                        self.printRow(label, max_p_wsd_label,
                                      "P WSD", stat["p_wsd"], end)
                    elif (r == 11):
                        self.printRow(label, max_p_kld_label,
                                      "P KLD", stat["p_kld"], end)
                    if (fits is not None):
                        fit = fits[map, label]
                        if (r == 13):
                            pobj = self.getFitDistribution(fit)
                            print(f'{"Fit: ":>15}', end="")
                            print(f'{pobj:<15}', end=end)
                        elif (r == 14):
                            lobj, vvobj = self.getFitParam(fit, 0)
                            if (lobj is not None and vvobj is not None):
                                print(f'{lobj+": ":>15}', end="")
                                print(f'{"{:0.6f}".format(vvobj):<15}', end=end)
                        elif (r == 15):
                            lobj, vvobj = self.getFitParam(fit, 1)
                            if (lobj is not None and vvobj is not None):
                                print(f'{lobj+": ":>15}', end="")
                                print(f'{"{:0.6f}".format(vvobj):<15}', end=end)
                        elif (r == 16):
                            lobj, vvobj = self.getFitParam(fit, 2)
                            if (lobj is not None and vvobj is not None):
                                print(f'{lobj+": ":>15}', end="")
                                print(f'{"{:0.6f}".format(vvobj):<15}', end=end)
            if not loopdone:
                print("")
                print("")

    def getFitDistribution(self, fit):
        if (fit["best"] is not None):
            key_object = list(fit["best"].keys())
            if (len(key_object) == 1):
                print_object = key_object[0]
            else:
                print_object = "n/a"
        else:
            print_object = "n/a"
        return print_object

    def getFitParam(self, fit, n):
        if (fit["best"] is not None):
            key_object = list(fit["best"].keys())
            if (len(key_object) == 1):
                value_object = fit["best"][key_object[0]]
                value_key_object = list(value_object.keys())
                if (n < len(value_key_object)):
                    label = value_key_object[n]
                    value = value_object[value_key_object[n]]
                else:
                    label = None
                    value = None
            else:
                label = None
                value = None
        else:
            label = None
            value = None
        return label, value

    # ██╗  ██╗███████╗██╗     ██████╗ ███████╗██████╗ ███████╗
    # ██║  ██║██╔════╝██║     ██╔══██╗██╔════╝██╔══██╗██╔════╝
    # ███████║█████╗  ██║     ██████╔╝█████╗  ██████╔╝███████╗
    # ██╔══██║██╔══╝  ██║     ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║
    # ██║  ██║███████╗███████╗██║     ███████╗██║  ██║███████║
    # ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝

    def getMatchesAndOthers(self, query, textQuery, map, batch):
        # cosine distances
        distances = self.getDistances(query, batch)
        qidx = np.argwhere(self.usedClassNames == textQuery)
        qlabel = self.usedClasses[qidx]
        qlabel = int(qlabel)
        map = int(map)

        if (map != -1):
            matched = self.getDistParam(distances, qlabel, map)
            others = self.getDistParam(distances, qlabel, map, True)
        else:
            matched = self.getDistParamAllMaps(distances, qlabel)
            others = self.getDistParamAllMaps(distances, qlabel, True)
        return matched, others

    def getMean(self, vecs):
        return np.mean(vecs, axis=0)

    # Disable
    def disablePrint(self):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    # Restore
    def enablePrint(self):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # stat print settings
    def setPrintColumns(self, c):
        if (c <= 1):
            print("number of columns must be > 1")
        else:
            self.printColumns = c

    def setMatchHistogram(self, set):
        self.matchHistogram = set

    def setDisplayFits(self, set):
        self.displayFits = set

    def setHistogramSettings(self, mn, mx, bins):
        self.bmin = mn
        self.bmax = mx
        self.nbins = bins

    def setRelativeHistogram(self, set):
        self.relativeHistogram = set

    # stat print settings
    def setSelectedClasses(self, used):
        self.selectedClasses = np.copy(self.usedClasses)
        self.selectedClassNames = np.copy(self.usedClassNames)
        for u in used:
            select = used[u]
            if (not select):
                i = np.where(self.selectedClassNames == u)
                self.selectedClassNames = np.delete(self.selectedClassNames, i)
                self.selectedClasses = np.delete(self.selectedClasses, i)

    # get all distances
    def getDistances(self, query, batch):
        # return stored if query is the same
        if (self.got_distances and (self.d_query == query).all()):
            return self.distances

        embeddings = self.data[:, :, 3, :]
        orig_shape = np.copy((embeddings.shape[0], embeddings.shape[1]))
        embeddings = embeddings.reshape(-1, self.embeddingSize)
        distances = np.full((embeddings.shape[0]), -1, dtype=np.float32)
        for i in tqdm(range(embeddings.shape[0]), disable=batch, desc="Distances", leave=False):
            e = embeddings[i, :]
            d = self.distance(e, query)
            if (d > 1):
                print("distance anomaly:", d)
            distances[i] = d
        distances = distances.reshape(orig_shape)

        # store results
        self.distances = distances
        self.got_distances = True
        self.d_query = query

        return distances

    # gets data with the same label
    def getLabelData(self, l, map=0, instance=None, inverse_instance=False):
        mapdata = self.data[map, :, :, :]
        idx = mapdata[:, 0, 0] == l
        labelData = mapdata[idx]

        if(instance):
            if(inverse_instance):
                idx = labelData[:, 1, 0] != instance
            else:
                idx = labelData[:, 1, 0] == instance
            labelData = labelData[idx]

        return labelData

    def getAllEmbeddings(self, map):
        mapdata = self.data[map, :, :, :]
        embeddings = mapdata[:, 3, :]
        return embeddings

    def getEmbeddings(self, label, map, instance=None, inverse_instance=False):
        allFields = self.getLabelData(label, map, instance, inverse_instance)
        embeddings = allFields[:, 3, :]
        return embeddings

    def getDist(self, label, map, instance=None, inverse_instance=False):
        if (int(label) == int(self.aggregateLabel)):
            allFields = self.data[map, :, :, :]
        else:
            allFields = self.getLabelData(label, map, instance, inverse_instance)
        dist = allFields[:, 4, 0]
        return dist

    def getDistParam(self, distances, label, map, invert=False):
        mapdata = self.data[map, :, :, :]
        if (invert):
            idx = (mapdata[:, 0, 0] != label).reshape((-1))
        else:
            idx = (mapdata[:, 0, 0] == label).reshape((-1))
        if (int(label) == int(self.aggregateLabel)):
            idx = mapdata[:, 0, 0]

        dist = distances[map, idx]
        return dist

    def getDistParamAllMaps(self, distances, label, invert=False):
        mapdata = self.data[:, :, :, :]
        if (invert):
            idx = (mapdata[:, :, 0, 0] != label)
        else:
            idx = (mapdata[:, :, 0, 0] == label)
        if (int(label) == int(self.aggregateLabel)):
            idx = mapdata[:, :, 0, 0]

        dist = distances[idx]
        dist = dist.reshape(-1)
        return dist

    def getNameForLabel(self, label):
        idx = np.where(self.classes == label)
        name = self.classNames[idx]
        return name.item()

    def getLabelForName(self, name):
        idx = np.where(self.classNames == name)
        label = self.classes[idx]
        return label.item()

    def getInstances(self, label, map):
        labelData = self.getLabelData(label, map)
        instances = np.unique(labelData[:,1,0])
        return instances

    # ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ██╗ ██████╗
    # ██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗  ██║██╔════╝
    # ██████╔╝███████║██████╔╝███████╗██║██╔██╗ ██║██║  ███╗
    # ██╔═══╝ ██╔══██║██╔══██╗╚════██║██║██║╚██╗██║██║   ██║
    # ██║     ██║  ██║██║  ██║███████║██║██║ ╚████║╚██████╔╝
    # ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝

    def parseNumpy(self, input, append=False):
        data = np.load(input)
        count = data.shape[0]

        if (not self.dataInitialized or not append):
            self.data = np.full(
                (1, count, self.columns, self.embeddingSize), -1, dtype=np.float32)
            self.data[self.mapCount, :, :4, :] = data
            self.dataInitialized = True
        else:
            ocount = count
            if (self.data.shape[1] > count):
                count = self.data.shape[1]
            newRange = np.full(
                (count, self.columns, self.embeddingSize), -1, dtype=np.float32)
            newRange[:ocount, :4, :] = data
            self.data = np.pad(
                self.data, ((0, 1), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
            if count > self.data.shape[1]:
                self.data = np.pad(self.data, ((
                    0, 0), (0, count - self.data.shape[1]), (0, 0), (0, 0)), mode='constant', constant_values=0)
            self.data[self.mapCount, :, :, :] = newRange

        print("data shape", self.data.shape)
        self.mapCount += 1

    def parseCpp(self, input, append=False):
        # TODO
        return np.zeros((1000, 2, self.embeddingSize))

    def parseClassFile(self, file, delim):
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
# END CLASS

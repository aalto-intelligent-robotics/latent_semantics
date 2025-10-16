import argparse
import os
import time
from tqdm import tqdm
import pickle
import datetime
from joblib import Parallel, delayed
from pathlib import Path
import numpy as np
import os

# this is for vscode debugging
try:
    import sys
    sys.path.append("embeddings")
    sys.path.append("analysis")
    sys.path.append("analysis/test")
    sys.path.append("utils")
    from analyzer import EmbeddingAnalyzer
    from analyzer_tests import test_statistical_tests
    from metric_tests import test_metrics
    from embeddingCreator import EmbeddingCreator
    from logger import Logger, LogLevel
except:
    from analysis.analyzer import EmbeddingAnalyzer
    from analysis.test.analyzer_tests import test_statistical_tests
    from analysis.test.metric_tests import test_metrics
    from embeddings.embeddingCreator import EmbeddingCreator
    from utils.logger import Logger, LogLevel

# ███████╗██╗██╗     ███████╗    ██╗    ██╗ ██████╗
# ██╔════╝██║██║     ██╔════╝    ██║   ██╔╝██╔═══██╗
# █████╗  ██║██║     █████╗      ██║  ██╔╝ ██║   ██║
# ██╔══╝  ██║██║     ██╔══╝      ██║ ██╔╝  ██║   ██║
# ██║     ██║███████╗███████╗    ██║██╔╝   ╚██████╔╝
# ╚═╝     ╚═╝╚══════╝╚══════╝    ╚═╝╚═╝     ╚═════╝


def createDirs(file):
    path = os.path.dirname(os.path.abspath(file))
    Path(path).mkdir(parents=True, exist_ok=True)


def safeOpen(file, mode):
    createDirs(file)
    return open(file, mode)


def getFile(path, file, suffix):
    os.makedirs(path, exist_ok=True)
    fileName = path + "/" + file + "." + suffix
    file_exists = os.path.exists(fileName)
    if (file_exists):
        dt = datetime.datetime.now()
        dt_str = dt.strftime('_%d_%m_%Y_%H_%M_%S')
        fileName = path + "/" + file + dt_str + "." + suffix
    return fileName


def getHeader(long=False):
    if long:
        q = "queryName;queryMap;queryLabel;"
    else:
        q = "query;"
    return "name;map;label;" + q + "mean;std;min;max;b wsd;b kld;b jsd;b bha;p wsd;p kld;fit distribution;fit param 1;fit param 2;fit param 3;\n"


def addField(line, value):
    line += str(value) + ";"
    return line


def writeHeader(file):
    f = safeOpen(file, "w")
    line = getHeader()
    f.write(line)
    f.close()


def getStatLine(stat, fits, name, map, label, query=None, queryMap=None, queryName=None, queryLabel=None):
    line = ""
    line = addField(line, name)
    line = addField(line, map)
    line = addField(line, label)
    if (query != None):
        line = addField(line, query)
    else:
        line = addField(line, queryName)
        line = addField(line, queryMap)
        line = addField(line, queryLabel)
    line = addField(line, stat["mean"])
    line = addField(line, stat["std"])
    line = addField(line, stat["min"])
    line = addField(line, stat["max"])
    line = addField(line, stat["b_wsd"])
    line = addField(line, stat["b_kld"])
    line = addField(line, stat["b_jsd"])
    line = addField(line, stat["b_bha"])
    line = addField(line, stat["p_wsd"])
    line = addField(line, stat["p_kld"])
    # line = addField(line, stat["p_jsd"])
    # line = addField(line, stat["p_bha"])

    if (fits is not None):
        fit = fits[map, label]
        if (fit["best"] is not None):
            dist = analyzer.getFitDistribution(fit)
            line = addField(line, dist)
            _, vvobj = analyzer.getFitParam(fit, 0)
            if (vvobj is not None):
                line = addField(line, vvobj)
            _, vvobj = analyzer.getFitParam(fit, 1)
            if (vvobj is not None):
                line = addField(line, vvobj)
            _, vvobj = analyzer.getFitParam(fit, 2)
            if (vvobj is not None):
                line = addField(line, vvobj)

    line += "\n"
    return line


def writeStats(file, query, numMaps, labels, names, stats, fits):
    f = safeOpen(file, "a")

    for map in range(numMaps):
        idx = 0
        for label in labels:
            if (label == analyzer.aggregateLabel):
                continue
            name = names[idx]
            stat = stats[map, label]
            line = getStatLine(stat, fits, name, map, label, query)
            f.write(line)

            idx += 1
    f.close()

#  ██████╗██████╗  █████╗
# ██╔════╝██╔══██╗██╔══██╗
# ██║     ██║  ██║███████║
# ██║     ██║  ██║██╔══██║
# ╚██████╗██████╔╝██║  ██║
#  ╚═════╝╚═════╝ ╚═╝  ╚═╝


def writeCrossStatsHeader(file):
    f = safeOpen(file, "w")
    line = getHeader(True)
    f.write(line)
    f.close()


def writeCrossStatsLine(file, q_map, numMaps, labels, names, stats):
    f = safeOpen(file, "a")
    midx = 0
    for q_label in labels:
        if (q_label == analyzer.aggregateLabel):
            continue
        if (q_map, q_label) not in stats:
            continue
        stat = stats[q_map, q_label]
        q_name = names[midx]
        midx += 1

        for map in range(numMaps):
            idx = 0
            for label in labels:
                if (label == analyzer.aggregateLabel):
                    continue
                if (map, label) not in stat:
                    continue
                s = stat[map, label]
                name = names[idx]
                line = getStatLine(s, None, name, map, label,
                                   None, q_map, q_name, q_label)
                f.write(line)
                idx += 1
    f.close()


def writeCrossStats(file, numMaps, labels, names, stats):
    writeCrossStatsHeader(file)

    for q_map in range(numMaps):
        writeCrossStatsLine(file, q_map, numMaps, labels, names, stats)


# ██╗███╗   ██╗████████╗██████╗  █████╗
# ██║████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗
# ██║██╔██╗ ██║   ██║   ██████╔╝███████║
# ██║██║╚██╗██║   ██║   ██╔══██╗██╔══██║
# ██║██║ ╚████║   ██║   ██║  ██║██║  ██║
# ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝


def writeIntraHeader(file):
    f = safeOpen(file, "w")
    line = "map;label;distance;count;\n"
    f.write(line)
    f.close()

#NOW
def writeIntraLine(file, map, stats):
    f = safeOpen(file, "a")

    for label in stats:
        (dist,count) = stats[label]
        line = ""
        line = addField(line, map)
        line = addField(line, label)
        line = addField(line, dist)
        line = addField(line, count)
        line += "\n"
        f.write(line)
    f.close()

# ██╗    ██╗██╗  ██╗
# ██║    ██║╚██╗██╔╝
# ██║ █╗ ██║ ╚███╔╝
# ██║███╗██║ ██╔██╗
# ╚███╔███╔╝██╔╝ ██╗
#  ╚══╝╚══╝ ╚═╝  ╚═╝


def writeWilcoxonHeader(file):
    f = safeOpen(file, "a")
    f.write("query;test;statistic;pvalue;matchedembeddings;other embeddings;\n")
    f.close()


def writeWilcoxonLine(file, query, test, statistic, pvalue, match_size, others_size):
    f = safeOpen(file, "a")
    line = ""
    line = addField(line, query)
    line = addField(line, test)
    line = addField(line, statistic)
    line = addField(line, pvalue)
    line = addField(line, match_size)
    line = addField(line, others_size)
    line += "\n"
    f.write(line)
    f.close()

#  ██████╗██╗
# ██╔════╝██║
# ██║     ██║
# ██║     ██║
# ╚██████╗███████╗
#  ╚═════╝╚══════╝


def writeClassificationHeader(file):
    f = safeOpen(file, "a")
    f.write("map;label;name;accuracy;macro_averaged_precision;micro_averaged_precision;macro_averaged_recall;micro_averaged_recall;macro_averaged_f1;micro_averaged_f1;iou;\n")
    f.close()


def writeClassificationLine(file, cl):
    f = safeOpen(file, "a")
    for i in range(len(cl)):
        d = cl[i]
        if(d):
            line = ""
            line = addField(line, d[0])
            line = addField(line, d[1])
            line = addField(line, d[2])
            line = addField(line, d[3])
            line = addField(line, d[4])
            line = addField(line, d[5])
            line = addField(line, d[6])
            line = addField(line, d[7])
            line = addField(line, d[8])
            line = addField(line, d[9])
            line = addField(line, d[10])
            line += "\n"
            f.write(line)
    f.close()

# ██╗ █████╗
# ██║██╔══██╗
# ██║███████║
# ██║██╔══██║
# ██║██║  ██║
# ╚═╝╚═╝  ╚═╝


def writeInstanceHeader(file):
    f = safeOpen(file, "a")
    f.write("map;queryLabel;queryInstance;label;instance;p_wsd\n")
    f.close()


def writeInstanceLine(file, stats, mapCount, selectedClasses):
    f = safeOpen(file, "a")

    for map in range(mapCount):
        for q_label in selectedClasses:
            q_instances = analyzer.getInstances(q_label, map)
            for q_instance in q_instances:
                try:
                    q_stats = stats[map,q_label,q_instance]
                    for label in selectedClasses:
                        instances = analyzer.getInstances(label, map)
                        for instance in instances:
                            try:
                                metric = q_stats[label, instance]["p_wsd"]
                                line = ""
                                line = addField(line, map)
                                line = addField(line, q_label)
                                line = addField(line, q_instance)
                                line = addField(line, label)
                                line = addField(line, instance)
                                line = addField(line, metric)
                                line += "\n"
                                f.write(line)
                            except:
                                continue
                except:
                    continue
    f.close()

#  ██████╗███████╗
# ██╔════╝██╔════╝
# ██║     █████╗
# ██║     ██╔══╝
# ╚██████╗██║
#  ╚═════╝╚═╝


def writeConfusionMatrix(matrix_file, labels_file, matrix, labels):
    f = safeOpen(labels_file, "a")
    # header
    line = ""
    for label in labels:
        line = addField(line, label)
    line = line.rstrip(";")
    line += "\n"
    f.write(line)
    f.close()

    f = safeOpen(matrix_file, "a")

    line = ""
    for label in labels:
        line = addField(line, label)
    line = line.rstrip(";")
    line += "\n"
    f.write(line)

    # data
    for i in range(matrix.shape[1]):
        line = ""

        for j in range(matrix.shape[0]):
            line = addField(line, matrix[i,j])
        line = line.rstrip(";")
        line += "\n"
        f.write(line)

    f.close()

# ███╗   ███╗██████╗
# ████╗ ████║██╔══██╗
# ██╔████╔██║██████╔╝
# ██║╚██╔╝██║██╔═══╝
# ██║ ╚═╝ ██║██║
# ╚═╝     ╚═╝╚═╝


def mp_analysis(cls, query, analyzer):
    stats, fits = analyzer.analyze(
        query=query, fit=False, distanalysis=False, textQuery=cls, batch=True)
    return (stats, fits, cls)


# ▄ ██╗▄▄ ██╗▄▄ ██╗▄▄ ██╗▄
#  ████╗ ████╗ ████╗ ████╗
# ▀╚██╔▀▀╚██╔▀▀╚██╔▀▀╚██╔▀
#   ╚═╝   ╚═╝   ╚═╝   ╚═╝


# ███╗   ███╗ █████╗ ██╗███╗   ██╗
# ████╗ ████║██╔══██╗██║████╗  ██║
# ██╔████╔██║███████║██║██╔██╗ ██║
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝


# ▄ ██╗▄▄ ██╗▄▄ ██╗▄▄ ██╗▄
#  ████╗ ████╗ ████╗ ████╗
# ▀╚██╔▀▀╚██╔▀▀╚██╔▀▀╚██╔▀
#   ╚═╝   ╚═╝   ╚═╝   ╚═╝


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./analysis_tool.py")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--input', '-i',
        dest="input_file",
        type=str,
    )
    group.add_argument(
        '--dir', '-d',
        dest="multifile_dir",
        type=str,
    )
    group.add_argument(
        "--test", "-t",
        dest="test",
        action="store_true",
        default=False
    )
    parser.add_argument(
        '--numpy', '-n',
        dest="numpy",
        action="store_true",
        required=False,
        default=True
    )
    parser.add_argument(
        '--classes', '-c',
        dest="classes_path",
        type=str,
        required=False,
        default="cfg/mpcat40.tsv"
    )
    parser.add_argument(
        '--delimiter', '-dlm',
        dest="delimiter",
        type=str,
        required=False,
        default="\t"
    )
    parser.add_argument(
        '--embedding_size', '-es',
        dest="embedding_size",
        type=int,
        required=False,
        default=512
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Use cpu or cuda.",
    )
    parser.add_argument(
        "--batch", "-b",
        dest="batch",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--cross_dist_analysis", "-cda",
        dest="crossDistributionAnalysis",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--intra_map", "-iam",
        dest="intraMapAnalysis",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--statistic_tests", "-st",
        dest="statistic_tests",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--classify", "-cl",
        dest="classify",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--confusion", "-cf",
        dest="confusion",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--instance_analysis", "-ia",
        dest="instance_analysis",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--aggregate_classify", "-ac",
        dest="aggregate_classify",
        action="store_true",
        required=False,
        default=False
    )
    # parser.add_argument(
    #     "--aggregate_confusion", "-ac",
    #     dest="aggregate_confusion",
    #     action="store_true",
    #     required=False,
    #     default=False
    # )
    parser.add_argument(
        "--batch_out_dir", "-bo",
        dest="batch_out_dir",
        type=str,
        required=False,
        default=""
    )
    parser.add_argument(
        "--multiprocessing", "-mp",
        dest="use_multiprocessing",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--debug", "-dbg",
        dest="debug",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--debug_dir", "-dd",
        dest="debug_dir",
        type=str,
        required=False,
        default= os.environ['DATA_DIR'] + "/dbg"
    )
    parser.add_argument(
        "--mean", "-m",
        dest="mean",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--skip", "-sk",
        dest="skip",
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        "--intra_suffix", "-ix",
        dest="intra_suffix",
        type=str,
        required=False,
        default=""
    )
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    input_file = FLAGS.input_file
    multifile_dir = FLAGS.multifile_dir
    isNumpy = FLAGS.numpy
    embedding_size = FLAGS.embedding_size
    classes_path = FLAGS.classes_path
    delimiter = FLAGS.delimiter
    device = FLAGS.device
    batch = FLAGS.batch
    batch_out_dir = FLAGS.batch_out_dir
    crossDistributionAnalysis = FLAGS.crossDistributionAnalysis
    statistic_tests = FLAGS.statistic_tests
    use_multiprocessing = FLAGS.use_multiprocessing
    debug = FLAGS.debug
    debug_dir = FLAGS.debug_dir
    mean = FLAGS.mean
    skip = FLAGS.skip
    testmode = FLAGS.test
    classify = FLAGS.classify
    aggregate_classify = FLAGS.aggregate_classify
    instance_analysis = FLAGS.instance_analysis
    confusion = FLAGS.confusion
    intraMapAnalysis = FLAGS.intraMapAnalysis
    intra_suffix = FLAGS.intra_suffix
    # aggregate_confusion = FLAGS.aggregate_confusion

logger = Logger("analysis_tool", level=LogLevel.INFO, verbosity=LogLevel.NONE)

if (not use_multiprocessing):
    logger.info("note: not using multiprocessing!")


# create analyzer
if(testmode):
    analyzer = EmbeddingAnalyzer(input=None, dir=None, isNumpy=isNumpy, classes_path=None,
                             delimiter=delimiter, device=device, embeddingSize=embedding_size)
else:
    analyzer = EmbeddingAnalyzer(input=input_file, dir=multifile_dir, isNumpy=isNumpy, classes_path=classes_path,
                             delimiter=delimiter, device=device, embeddingSize=embedding_size)


if(not testmode):
    logger.info("Initializing...")
    creator = EmbeddingCreator(device)

# ██████╗  █████╗ ████████╗ ██████╗██╗  ██╗
# ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║  ██║
# ██████╔╝███████║   ██║   ██║     ███████║
# ██╔══██╗██╔══██║   ██║   ██║     ██╔══██║
# ██████╔╝██║  ██║   ██║   ╚██████╗██║  ██║
# ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝

if (batch or crossDistributionAnalysis or statistic_tests or classify or instance_analysis or confusion or intraMapAnalysis):
    if (not batch_out_dir):
        logger.info(
            "You must set batch_out_dir (--batch_out_dir, -bo) in batch mode!")
        exit()

    # classNames have all possible labels
    # usedClassNames have all existing labels and aggregate label
    # selectedClassNames can be changed, but by default it is the list of existing labels
    classes = analyzer.selectedClassNames
    if (input_file):
        logger.info("Batch processing", input_file)
    else:
        logger.info("Batch processing", multifile_dir)

    # create queries faster
    if (batch or statistic_tests):
        queries = []
        for i in tqdm(range(len(classes)), leave=False):
            queries.append(creator.get_text_embedding(classes[i]))

    # histogram analysis
    if (batch):
        batch_out_file = getFile(batch_out_dir, "batch", "out")

        writeHeader(batch_out_file)

        start = time.time()
        if (use_multiprocessing):
            results = Parallel(n_jobs=1)(delayed(mp_analysis)(
                classes[i], queries[i], analyzer) for i in tqdm(range(len(classes))))

            dt = datetime.datetime.now()
            dt_str = dt.strftime('_%d_%m_%Y_%H_%M_%S')
            if (debug):
                with safeOpen(debug_dir+'/dbg-results-'+dt_str+'.pkl', 'wb') as f:
                    pickle.dump(results, f)

            for result in results:
                (stats, fits, cls) = result
                writeStats(batch_out_file, cls, analyzer.mapCount,
                           analyzer.usedClasses, analyzer.usedClassNames, stats, fits)
        else:
            results = {}
            for i in tqdm(range(len(classes))):
                cls = classes[i]
                query = queries[i]
                stats, fits = analyzer.analyze(
                    query=query, fit=False, distanalysis=False, textQuery=cls, batch=True)
                writeStats(batch_out_file, cls, analyzer.mapCount,
                           analyzer.usedClasses, analyzer.usedClassNames, stats, fits)

        end = time.time()
        dur = end - start
        logger.info("took", dur, "s")
        logger.info("Batch processing done. Wrote output to file",
                    batch_out_file)

    # crossdistribution analysis
    if (crossDistributionAnalysis):
        cda_out_file = getFile(batch_out_dir, "stats", "out")
        writeCrossStatsHeader(cda_out_file)

        start = time.time()
        allStats = {}
        midx = 0

        for map in tqdm(range(skip, analyzer.mapCount), disable=False, leave=False, desc="cross-distribution analysis: map"):
            if (use_multiprocessing):
                partialStats = analyzer.mp_crossDistributionAnalysisSingleMap(
                    map=map, batch=True, numProcess=8)
            else:
                partialStats = analyzer.crossDistributionAnalysisSingleMap(
                    map=map, batch=True)

            for (m, l) in partialStats:
                allStats[m, l] = partialStats[m, l]
            if (debug):
                with safeOpen(debug_dir+'/dbg-distStats-'+str(map)+'.pkl', 'wb') as f:
                    pickle.dump(partialStats, f)

            writeCrossStatsLine(cda_out_file, map, analyzer.mapCount,
                                analyzer.usedClasses, analyzer.usedClassNames, partialStats)
            midx += 1
            logger.info("Written map data", midx, "/", analyzer.mapCount)
        end = time.time()
        dur = end - start
        logger.info("took", dur, "s")

    # intra-map analysis
    # NOW
    if (intraMapAnalysis):
        iam_out_file = getFile(batch_out_dir, "intramap"+str(intra_suffix), "out")
        writeIntraHeader(iam_out_file)

        start = time.time()

        midx = 0

        for map in tqdm(range(skip, analyzer.mapCount), disable=False, leave=False, desc="intra-map analysis: map"):
            partialStats = analyzer.intraMapAnalysis(map=map, batch=True)

            writeIntraLine(iam_out_file, map, partialStats)
            midx += 1
            logger.info("Written map data", midx, "/", analyzer.mapCount)
        end = time.time()
        dur = end - start
        logger.info("took", dur, "s")

    # wilcoxon histogram analysis
    if (statistic_tests):
        st_out_file = getFile(batch_out_dir, "statistic_tests", "out")
        writeWilcoxonHeader(st_out_file)
        for i in tqdm(range(len(classes))):
            cls = classes[i]
            query = queries[i]
            wx, we, ks, kw, mw, td = analyzer.statistical_tests(
                query, textQuery=cls, map=-1, batch=True)
            writeWilcoxonLine(st_out_file, cls, "wilcoxon", wx[0], wx[1], wx[2], wx[3])
            writeWilcoxonLine(st_out_file, cls, "welch", we[0], we[1], we[2], we[3])
            writeWilcoxonLine(st_out_file, cls, "kolmogorov", ks[0], ks[1], ks[2], ks[3])
            writeWilcoxonLine(st_out_file, cls, "kruskal", kw[0], kw[1], kw[2], kw[3])
            writeWilcoxonLine(st_out_file, cls, "mann", mw[0], mw[1], mw[2], mw[3])
            writeWilcoxonLine(st_out_file, cls, "tukey", td[0], td[1], td[2], td[3])

    # classification
    if(classify):
        names = analyzer.classNames
        labels = analyzer.classes
        queries = []
        for i in tqdm(range(len(names)), leave=False):
            queries.append(creator.get_text_embedding(names[i]))
        output = analyzer.classification(
            queries=queries, labels=labels, aggregate=aggregate_classify, batch=True)
        cl_out_file = getFile(batch_out_dir, "classification", "out")
        writeClassificationHeader(cl_out_file)
        writeClassificationLine(cl_out_file, output)

    # confusion matrix
    if (confusion):
        names = analyzer.classNames
        labels = analyzer.classes
        queries = []
        for i in tqdm(range(len(names)), leave=False):
            queries.append(creator.get_text_embedding(names[i]))
        matrix, labels = analyzer.confusion(queries, labels)
        matrix_out_file = getFile(batch_out_dir, "confusion_matrix", "out")
        labels_out_file = getFile(batch_out_dir, "confusion_labels", "out")
        writeConfusionMatrix(matrix_out_file, labels_out_file, matrix, labels)

    # instance analysis
    if(instance_analysis):
        names = analyzer.classNames
        for name in names:
            output, mapcount, selectedClasses = analyzer.instance_statistics(textQuery=name, batch=True)
            ia_out_file = getFile(batch_out_dir, "instance", "out")
            writeInstanceHeader(ia_out_file)
            writeInstanceLine(ia_out_file, output, mapcount, selectedClasses)
    exit()

# ██████╗ ███████╗ ██████╗ ██╗   ██╗██╗      █████╗ ██████╗
# ██╔══██╗██╔════╝██╔════╝ ██║   ██║██║     ██╔══██╗██╔══██╗
# ██████╔╝█████╗  ██║  ███╗██║   ██║██║     ███████║██████╔╝
# ██╔══██╗██╔══╝  ██║   ██║██║   ██║██║     ██╔══██║██╔══██╗
# ██║  ██║███████╗╚██████╔╝╚██████╔╝███████╗██║  ██║██║  ██║
# ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝


# MAIN LOOP
inp = ''
fit = False
metrics = False

while (inp != 'q'):
    print(" ")
    print("Analysis tool")
    print("*"*40)
    print("Settings:")
    if (input_file):
        print("Reading:", input_file)
    else:
        print("Reading:", multifile_dir)
    print("Fit distributions:", fit)
    print("Calculate metrics:", metrics)
    print("Display fits:", analyzer.displayFits)
    print("Columns:", analyzer.printColumns)
    print("Histogram:", analyzer.bmin, "-",
          analyzer.bmax, ": ", analyzer.nbins, "bins")
    print("Relative histogram:", analyzer.relativeHistogram)
    print("Match histogram:", analyzer.matchHistogram)
    print("*"*40)
    print("t: query text")
    print("i: query image")
    print("n: instance statistics")
    print("c: classification")
    print("v: confusion matrix")
    print("d: cross-distribution analysis")
    print("a: intra-map analysis")
    print("-"*10)
    print("1: Wilcoxon (binned) signed rank test")
    print("2: Welch's t-test")
    print("3: Kolmogorov-Smirnov test")
    print("4: Kruskal-Wallis test")
    print("5: Mann-Whitney U test")
    print("6: Tukey-Duckworth test")
    print("0: all tests")
    print("-"*10)
    print("o: test statistical tests")
    print("l: test metrics")
    print("-"*10)
    print("s: select classes")
    print("f: change fit mode")
    print("g: change metric calculation")
    print("p: display fits")
    print("c: change # columns")
    print("h: histogram settings")
    print("m: change match histogram")
    print("r: change relative histogram")
    print("b: print instance statistics")
    print("-"*10)
    print("q: quit")
    inp = input(": ")
    gotQuery = False
    ############################################################################
    # STANDARD QUERIES
    ############################################################################
    if (inp == 't'):
        inp = input("Input text query: ")
        textQuery = inp
        query = creator.get_text_embedding(inp)
        gotQuery = True
    ############################################################################
    elif (inp == 'i'):
        textQuery = None
        inp = input("Input image query path: ")
        try:
            query = creator.get_image_embedding(inp)
            gotQuery = True
        except:
            print("bad image file/path")
    ############################################################################
    # Instances
    ############################################################################
    elif (inp == 'n'):
        inp = input("Selected class: ")
        textQuery = inp
        analyzer.instance_statistics(textQuery)
    ############################################################################
    # CDA
    ############################################################################
    elif (inp == 'd'):
        analyzer.crossDistributionAnalysis()
    ############################################################################
    # intra
    ############################################################################
    #NOW
    elif (inp == 'a'):
        m = 0
        dists = analyzer.intraMapAnalysis(m)
        for key in dists:
            (d,c) = dists[key]
            print(m, key, d, c)

    ############################################################################
    # CLASSIFICATION
    ############################################################################
    elif (inp == 'c'):
        names = analyzer.classNames
        labels = analyzer.classes

        queries = []
        for i in tqdm(range(len(names)), leave=False):
            queries.append(creator.get_text_embedding(names[i]))
        analyzer.classification(queries, labels)
    ############################################################################
    # CONFUSION
    ############################################################################
    elif (inp == 'v'):
        names = analyzer.classNames
        labels = analyzer.classes

        queries = []
        for i in tqdm(range(len(names)), leave=False):
            queries.append(creator.get_text_embedding(names[i]))
        matrix, labels = analyzer.confusion(queries, labels)

        matrix_out_file = getFile(os.environ['VLMAPS_DIR'] + "/data/mapdata/analysis", "confusion_matrix", "out")
        labels_out_file = getFile(os.environ['VLMAPS_DIR'] + "/data/mapdata/analysis", "confusion_labels", "out")
        writeConfusionMatrix(matrix_out_file, labels_out_file, matrix, labels)

    ############################################################################
    # STATS TESTS
    ############################################################################
    elif (inp == '1' or inp == '2' or inp == '3' or inp == '4' or inp == '5' or inp == '6' or inp == '7' or inp == '0'):
        orig = inp
        inp = input("Input text query: ")
        textQuery = inp
        query = creator.get_text_embedding(inp)
        inp = input("Which map (-1 = all):")
        try:
            map = int(inp)
        except:
            map = -1

        if (orig == '1'):
            analyzer.wilcoxon(query, textQuery, map)
        elif (orig == '2'):
            analyzer.welch(query, textQuery, map)
        elif (orig == '3'):
            analyzer.kolmogorov(query, textQuery, map)
        elif (orig == '4'):
            analyzer.kruskal(query, textQuery, map)
        elif (orig == '5'):
            analyzer.whitney(query, textQuery, map)
        elif (orig == '6'):
            analyzer.tukey(query, textQuery, map)
        elif (orig == '0'):
            analyzer.statistical_tests(query, textQuery, map)
        else:
            pass
    ############################################################################
    elif(inp == 'o'):
        snum = input("number of tests:")
        try:
            num = int(snum)
        except:
            num = -1
        test_statistical_tests(analyzer, num)
    ############################################################################
    elif (inp == 'l'):
        # snum = input("number of tests:")
        # try:
        #     num = int(snum)
        # except:
        #     num = -1
        outFile = "data/metrics_test.out"
        test_metrics(analyzer, outFile, 1)
    ############################################################################
    # settings
    ############################################################################
    elif (inp == 'f'):
        fit = not fit
    ############################################################################
    elif (inp == 'g'):
        metrics = not metrics
    ############################################################################
    elif (inp == 'p'):
        if (not analyzer.displayFits and not fit):
            fit = not fit
        analyzer.setDisplayFits(not analyzer.displayFits)
    ############################################################################
    elif (inp == 'c'):
        inp = input("New number of columns: ")
        c = int(inp)
        analyzer.setPrintColumns(c)
    ############################################################################
    elif (inp == 'm'):
        analyzer.setMatchHistogram(not analyzer.matchHistogram)
    ############################################################################
    elif (inp == 'r'):
        analyzer.setRelativeHistogram(not analyzer.relativeHistogram)
    ############################################################################
    elif (inp == 'b'):
        analyzer.printInstanceStatistics()
    ############################################################################
    elif (inp == 'h'):
        try:
            mn = float(input("Min: "))
            mx = float(input("Max: "))
            bins = int(input("Bins: "))
            analyzer.setHistogramSettings(mn, mx, bins)
        except:
            print("Bad input")
    ############################################################################
    elif (inp == 's'):
        print("Used classes:")
        used = {}
        idx = 0
        for cls in analyzer.usedClassNames:
            used[cls] = cls in analyzer.selectedClassNames
            if (used[cls]):
                print("x " + str(idx) + ": " + cls)
            else:
                print(" " + str(idx) + ": " + cls)
            idx += 1
        print("Input class number do invert selection")
        print("a: to select all")
        print("n: to select none")
        print("q: when done")

        inp = ''
        while (inp != 'q'):
            inp = input(": ")
            if (inp == 'a'):
                for cls in analyzer.usedClassNames:
                    used[cls] = True
            elif (inp == 'n'):
                for cls in analyzer.usedClassNames:
                    used[cls] = False
            else:
                try:
                    num = int(inp)
                    if (num < len(analyzer.usedClassNames)):
                        cls = analyzer.usedClassNames[num]
                        used[cls] = not used[cls]
                    else:
                        print("number out of range")
                except:
                    print("unknown input")

            idx = 0
            for cls in analyzer.usedClassNames:
                if (used[cls]):
                    print("x " + str(idx) + ": " + cls)
                else:
                    print("  " + str(idx) + ": " + cls)
                idx += 1
        analyzer.setSelectedClasses(used)
        inp = ''
    ############################################################################

    if (gotQuery):
        print("Querying", inp)
        analyzer.analyze(query=query, fit=fit, distanalysis=metrics,
                         textQuery=textQuery, batch=False, mean=mean)

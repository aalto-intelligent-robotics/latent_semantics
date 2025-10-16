import os
from pathlib import Path
import datetime


def createDirs(file):
    path = os.path.dirname(os.path.abspath(file))
    Path(path).mkdir(parents=True, exist_ok=True)


def safeOpen(file, mode):
    createDirs(file)
    return open(file, mode)


def writeClassificationHeader(file):
    f = safeOpen(file, "a")
    f.write("map;method;accuracy;macro_averaged_precision;micro_averaged_precision;macro_averaged_recall;micro_averaged_recall;macro_averaged_f1;micro_averaged_f1;macro_iou;micro_iou;macro_specificity;micro_specificity;macro_NPV;micro_NPV;macro_balanced_accuracy;micro_balanced_accuracy\n")
    f.close()


def getFile(path, file, suffix):
    os.makedirs(path, exist_ok=True)
    fileName = path + "/" + file + "." + suffix
    file_exists = os.path.exists(fileName)
    if (file_exists):
        dt = datetime.datetime.now()
        dt_str = dt.strftime('_%d_%m_%Y_%H_%M_%S')
        fileName = path + "/" + file + dt_str + "." + suffix
    return fileName


def addField(line, value):
    line += str(value) + ";"
    return line


def writeClassificationLine(file, cl):
    f = safeOpen(file, "a")
    for i in range(len(cl)):
        d = cl[i]
        if (d):
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
            line = addField(line, d[11])
            line = addField(line, d[12])
            line = addField(line, d[13])
            line = addField(line, d[14])
            line = addField(line, d[15])
            line = addField(line, d[16])
            line = line.rstrip(";")
            line += "\n"
            f.write(line)
    f.close()


def writeString(file, str, appendBreak=True):
    f = safeOpen(file, "a")
    if (appendBreak):
        str += "\n"
    f.write(str)
    f.close()

import numpy as np

class UniqueChecker:
    def __init__(self, name = ""):
        self.name = name
        self.sizes = []
        self.uniques = np.array([])

    def add(self, data):
        uniques = np.unique(data)
        self.sizes.append(uniques.size)
        self.uniques = np.concatenate((self.uniques, uniques))

    def eval(self):
        size = np.array(self.sizes)
        self.minUniques = np.min(size)
        self.maxUniques = np.max(size)
        self.avgUniques = np.mean(size)
        self.uniques = np.unique(self.uniques)

    def print(self):
        print("*"*80)
        if(self.name):
            print(self.name + " unique labels:")
        else:
            print("Unique labels:")
        print("min", self.minUniques)
        print("max", self.maxUniques)
        print("avg", self.avgUniques)
        print(self.uniques)
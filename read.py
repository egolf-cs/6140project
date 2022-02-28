import csv

class DatPt:

    def __init__(self, raw):
        ## The following two fields are used to validate that the CSV
        ## was read properly
        # This is the list of raw data.
        self.raw = raw
        # This is the indicator list of length 6.
        # A 1 at index i indicates that heuristic i was the best
        self.rawy = [int(s) for s in self.raw[-6:]]

        # x is the feature vector
        self.x = [float(s) for s in self.raw[:-6]]
        # y is the label of the feature used 0,1,..,5
        self.y = self.rawy.index(1)

    def validate(self):
        b1 = (len(self.rawy) == 6)
        b2 = (len(self.raw) == 57)
        b3 = ((len(self.x) + len(self.rawy)) == len(self.raw))
        b4 = self.rawy[self.y] == 1
        tmp = list(5*[-1])
        tmp.insert(self.y, 1)
        b5 = (tmp == self.rawy)
        return b1 and b2 and b3 and b4 and b5

class DatPts:

    def __init__(self, csv):
        self.ps = self.read_dat(csv)
        self.name = csv
        self.xs = [p.x for p in self.ps]
        self.ys = [p.y for p in self.ps]

    def read_dat(self, file):
        res = []
        with open(file, newline='') as csvfile:
            rs = csv.reader(csvfile, delimiter=',')
            for r in rs:
                p = DatPt(r)
                res += [p]
        return res

    # ps is a list of DatPt's
    def validate(self):
        v = [p.validate() for p in self.ps]
        return (False not in v)

    def __repr__(self):
        return f"<DatPts '{self.name}' len={len(self.ps)}>"

    def concat(self, dps):
        self.ps = self.ps + dps.ps
        self.name = f"{self.name},{dps.name}"
        self.xs = [p.x for p in self.ps]
        self.ys = [p.y for p in self.ps]

import csv

class DatPt:

    def __init__(self, raw):
        ## The following two fields are used to validate that the CSV
        ## was read properly
        # This is the list of raw data.
        self.raw = raw
        # This is the indicator list of length 6.
        # A 1 at index i indicates that heuristic i was the best
        self.rawy = [float(s) for s in self.raw[-5:]]

        # x is the feature vector
        self.rawx = [float(s) for s in self.raw[:-5]]
        self.x = self.rawx[:4] + self.rawx[5:34] + self.rawx[35:]
        # y is the label of the feature used 0,1,..,5
        z = list(zip(range(len(self.rawy)),self.rawy))
        z = list(filter(lambda x: x[1] != -100, z))
        if z == []:
            self.y = 5
        else:
            tmp = min(z, key=lambda x: x[1])
            self.y = tmp[0]

    def validate(self):
        assert(self.rawx[4] == 0)
        assert(self.rawx[34] == 0)
        assert(len(self.rawx) == 53)
        assert(len(self.x) == 51)
        assert(len(self.rawy) == 5)
        if self.y == 5:
            self.rawy = 5*[-100]
        else:
            tmp1 = self.rawy[self.y]
            tmp2 = list(filter(lambda x: x != -100, self.rawy))
            assert(tmp1 == min(tmp2))

class DatPts:

    def __init__(self, csv=""):
        if csv != "":
            self.ps = self.read_dat(csv)
            self.name = csv
            self.xs = [p.x for p in self.ps]
            self.ys = [p.y for p in self.ps]
            self.rawys = [p.rawy for p in self.ps]

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
        [p.validate() for p in self.ps]

    def __repr__(self):
        return f"<DatPts '{self.name}' len={len(self.ps)}>"

    def normalize(self, is_train, scaler):
        if is_train:
            scaler.fit(self.xs)
            self.xs = scaler.transform(self.xs)
        else:
            self.xs = scaler.transform(self.xs)
        return scaler

    def concat(self, dps):
        self.ps = self.ps + dps.ps
        self.name = f"{self.name},{dps.name}"
        self.xs = [p.x for p in self.ps]
        self.ys = [p.y for p in self.ps]

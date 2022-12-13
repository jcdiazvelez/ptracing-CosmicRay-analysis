import numpy as np


class PathSegment:
    def __init__(self, datum):
        self.trackId = int(datum[0])
        self.time = datum[1]
        self.x = datum[2]
        self.y = datum[3]
        self.z = datum[4]
        self.r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        self.px = datum[5]
        self.py = datum[6]
        self.pz = datum[7]
        self.p = np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        self.pid = int(datum[8])
        self.weight = datum[9]
        try:
            self.Bx = datum[10]
            self.By = datum[11]
            self.Bz = datum[12]
            self.B = np.sqrt(self.Bx ** 2 + self.By ** 2 + self.Bz ** 2)
            self.status = datum[13]
        except:
            self.Bx = 0
            self.By = 0
            self.Bz = 0
            self.B = 1
            self.status = None

    def __repr__(self):
        s = "track: %s\n" % self.trackId
        s += "\t time: %s\n" % self.time
        s += "\t position: (%s,%s,%s)\n" % (self.x, self.y, self.z)
        s += "\t momentum: (%s,%s,%s)\n" % (self.px, self.py, self.pz)
        s += "\t distance from the Sun: (%s)\n" % self.r
        s += "\t PID: %s\n" % self.pid
        s += "\t weight: %s" % self.weight
        return s

    def pitch_angle(self):
        return np.arccos((self.px * self.Bx + self.py * self.By + self.pz * self.Bz) / (self.p * self.B))

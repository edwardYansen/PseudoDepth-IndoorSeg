
class DepthConstSet:
    def __init__(self):
        self.raw = "raw"
        self.marigold = "marigold"
        self.midas = "midas"
        self.depthAnythingV2 = "depthAnythingV2"
        self.HHA = "HHA"

    def get_all(self):
        return {
            "raw": self.raw,
            "marigold": self.marigold,
            "midas": self.midas,
            "depthAnythingV2": self.depthAnythingV2,
            "HHA": self.HHA
        }
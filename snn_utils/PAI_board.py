import os
from torch import Tensor
from spike_tensor import SpikeTensor
from runtime import *
from utils import *
from Ethernet_utils.frameHandler import FrameHandler

import time
import numpy as np

class PAIBoard(object):
    def __init__(self, baseDir: str, timestep: int):
        self._dir = baseDir
        self._ts = timestep

        self.frameFormats, self.frameNums, self.inputNames = loadInputFormats(self._dir)
        self.outDict, self.shapeDict, self.scaleDict, self.mapper = loadOutputFormat(self._dir)

        auxNetDir = os.path.join(self._dir, "auxNet")
        self.preNet, self.postNet = loadAuxNet(auxNetDir)

        self.inputFramePath = os.path.join("./files", "input.txt") # TODO: remove files, directly 
        self.outputFramePath = os.path.join("./files", "out.txt")

        # self.ethernet = FrameHandler()

    def connect(self):
        # self.ethernet.ethernetHandler.reconnect()
        None

    def __call__(self, x: Tensor) -> Tensor:
        format = np.array(self.frameFormats[2:-1])
        # tensor2frame
        t0 = time.time()
        for i in range(10):
            tt0 = time.time()
            data = x.unsqueeze(0).expand(self._ts, *x.shape)
            data = SpikeTensor(data, self._ts, 1)
            dataDict = runPreNet(self.preNet, self.inputNames, *[data])
            tt1 = time.time()
            print(tt1 - tt0)
            encodeDataFrame(dataDict, self.frameFormats, self.frameNums, self.inputNames, self.inputFramePath, format)
        
        t1 = time.time()
        print((t1 - t0) / 10)
        # send to Ethernet
        # self.ethernet.transformFrameFile("input") 
        # self.ethernet.writeWithGapFrameNum("input", 10000, 1)     
        # receive from Ethernet
        # self.ethernet.read()
        # frame2tensor
        # dataFrames = getData(self.outputFramePath)
        # outDataDict = decodeDataFrame(dataFrames, self.outDict, self.shapeDict, self.scaleDict, self.mapper, self._ts)
        # # if one output
        # for name in outDataDict.keys():
        #     return outDataDict[name]
    
import torch
import numpy as np
import os
import json
import pickle
import time
from copy import deepcopy

from hardware import Hardware
from spike_tensor import SpikeTensor

class MASK:
    HEADBEGIN = 60
    CHIPBEGIN = 50
    CHIPMASK = ((1 << 10) - 1)
    COREBEGIN = 40
    COREMASK = ((1 << 10) - 1)
    STARBEGIN = 30
    STARMASK = ((1 << 10) - 1)
    AXONBEGIN = 16
    AXONMASK = ((1 << 11) - 1)
    SLOTBEGIN = 8
    SLOTMASK = ((1 << 8) - 1)
    DATABEGIN = 0
    DATAMASK = ((1 << 8) - 1)

def CHIPID(spike):
    assert spike < (3 << 62), spike
    return (spike >> MASK.CHIPBEGIN) & MASK.CHIPMASK
def COREID(spike):
    return (spike >> MASK.COREBEGIN) & MASK.COREMASK
def STARID(spike):
    return (spike >> MASK.STARBEGIN) & MASK.STARMASK
def AXONID(spike):
    return (spike >> MASK.AXONBEGIN) & MASK.AXONMASK
def SLOTID(spike):
    return (spike >> MASK.SLOTBEGIN) & MASK.SLOTMASK
def DATAID(spike):
    return (spike >> MASK.DATABEGIN) & MASK.DATAMASK

def ISSYNC(spike):
    return (spike >> MASK.HEADBEGIN) == 9
def ISINIT(spike):
    return (spike >> MASK.HEADBEGIN) == 11

def runPreNet(preNet, inputNames, *data):
    if len(preNet.inputs_nodes) > 0:
        preNet(*data)
        return preNet.nodes
    else:
        outData = dict()
        for i in range(len(inputNames)):
            outData[inputNames[i]] = data[i]
        return outData

def runPostNet(postNet, dataDict):
    data = [dataDict[name] for name in postNet.inputs_nodes]
    if len(postNet.inputs_nodes) > 0:
        postNet(*data)
        return postNet.nodes
    else:
        return dataDict

def encodeDataFrame(dataDict, frameFormats, frameNums, nameLists, filePath, format):

    # with open(filePath,'w') as f:
    f = open(filePath,'w')

        # for init frames
    initFrames = "\n".join(frameFormats[:frameNums[0]]) + "\n"
    base = frameNums[0]
    writeInitFrames = initFrames[:65]
    f.write(writeInitFrames)

    # format = np.array(frameFormats[2:-1])
    # for data frames
    for name in nameLists:
        
        if hasattr(dataDict[name], "timesteps"):
            data = dataDict[name].data
            data = data.reshape(-1)
            isSNN = True
        else:
            data = dataDict[name].reshape(-1)
            isSNN = False

        np_data = np.array(data)
        index = np.nonzero(np_data)
        frames = format[index]
        for frame in frames:
            f.write(frame + "00000001\n")

        # for spike, number in zip(data, frameNums[1:]):
        #     if spike == 0:
        #         base += number
        #         continue
        #     if spike < 0:
        #         spike += (1 << 8)
        #     spike = int(spike)
        #     dataStr = "{:08b}\n".format(spike)
        #     dataStrframes = dataStr.join(frameFormats[base : base + number]) 
        #     f.write(dataStrframes+dataStr)
        #     base += number

    # for sync frames
    # syncFrames = "\n".join(frameFormats[base:]) + "\n"
    syncFrames = frameFormats[-1] + "\n"
    if (len(syncFrames) > 1):
        f.write(syncFrames)
    # print(t1 - t0)
    # f.close()

def decodeDataFrame(dataFrames, outputDict, shapeDict, scaleDict, mapper, timeStep):
    dataDict = dict()
    shapeLen = dict()
    timeSteps = dict()
    for name, shape in shapeDict.items():
        shapeLen[name] = np.prod(shape)
        dataDict[name] = torch.zeros(shapeLen[name] * timeStep)
        timeSteps[name] = 0
    hardwareAxonBit = Hardware.getAttr("AXONBIT", True)
    for frame in dataFrames:
        if frame == '':
            break
        pos = (int(frame[4:24],2) << hardwareAxonBit) + int(frame[37:48],2)
        data = int(frame[-8:],2)
        newTimeStep = int(frame[48:56],2)
        name, tensorPos = mapper[pos]
        if timeSteps[name] <= newTimeStep:
            timeSteps[name] = newTimeStep
        else:
            timeSteps[name] = (256 - timeSteps[name]) + newTimeStep
        dataDict[name][tensorPos + shapeLen[name] * timeSteps[name]] = data

    for name, outputs in outputDict.items():
        pos = 0
        if len(outputs) == 1 and outputs[0] == name:
            continue
        for output in outputs:
            dataDict[name][pos: (pos + shapeLen[output] * timeStep)] = dataDict[output][:]
            pos += shapeLen[output] * timeStep
    for name in outputDict.keys():
        dataDict[name] = dataDict[name].reshape(timeStep, *shapeDict[name])
    for name in outputDict.keys():
        shape = shapeDict[name]
        scale = np.array(scaleDict[name])
        dataDict[name] = dataDict[name].reshape(timeStep, *shape).mean(0) * scale
        # dataDict[name] = dataDict[name].reshape(timeStep, *shape)
    return dataDict

def loadInputFormats(baseDir):
    formatDir = os.path.join(baseDir, "formats")
    with open(os.path.join(formatDir, "formats.pkl"),"rb") as formatInputFile:
        inputFormats = pickle.load(formatInputFile)
    with open(os.path.join(formatDir, "numbers.pkl"),"rb") as numberFile:
        numbers = pickle.load(numberFile)
    with open(os.path.join(formatDir, "inputNames.pkl"),"rb") as nameInputFile:
        inputNames = pickle.load(nameInputFile)
    return inputFormats, numbers, inputNames

def genInputFrames(baseDir, inputMode, preNet, timeSteps):
    inputDir = os.path.join(baseDir, "input")
    frameDir = os.path.join(baseDir, "frames")
    frameFormats, frameNums, inputNames = loadInputFormats(baseDir)
    files = os.listdir(inputDir)
    inputs = list()
    for dataFile in files:
        if dataFile.startswith("."):
            continue
        fullFileName = os.path.join(inputDir, dataFile)
        with open(fullFileName, 'rb') as f:
            data = torch.load(f).float()
        
        data = data.unsqueeze(0).expand(timeSteps,*data.shape)
        if inputMode == 'snn':
            data = SpikeTensor(data, timeSteps, 1)
        else:
            data.scale = 1
        dataDict = runPreNet(preNet, inputNames, *[data])
        filePath = os.path.join(frameDir, dataFile)
        encodeDataFrame(dataDict, frameFormats, frameNums, inputNames, filePath)
    return

class Neuron:
    def __init__(self, chipId, coreId, neuronId, para, weights, bitWidth, LCN):
        self.tickReltive = para[0]
        # assert self.tickReltive == 0, f"{chipId}, {coreId}, {neuronId}, {para}"
        self.coreId = coreId
        self.neuronId = neuronId
        self.chipId = chipId
        destAxon = para[1]
        destCore = (para[2] << 5) + para[3]
        destStar = (para[4] << 5) + para[5]
        destChip = (para[6] << 5) + para[7]
        cores = {destCore}
        for i in range(10):
            if (destStar >> i) & 1:
                star = 1 << i
                tmp = deepcopy(cores)
                for c in tmp:
                    cores.add(c ^ star)
        
        self.spikeFormats = list()
        for core in cores:
            spikeFormat = \
                (8 << 60) + (destChip << 50) + (core << 40) + \
                + (destAxon << 16)
            assert spikeFormat < (3 << 62), f"{destChip},{destCore},{destStar},{destAxon}"
            self.spikeFormats.append(spikeFormat)
        self.resetMode = para[8]
        self.resetV = para[9]
        self.leakPost = para[10]
        self.threMaskCtrl = para[11]
        self.threNegMode = para[12]
        self.thresholdNeg = para[13]
        self.thresholdPos = para[14]
        self.leakReFlag = para[15]
        self.leakDetStoch = para[16]
        self.leakV = para[17]
        self.weightDetStoch = para[18]
        self.bitTrunc = para[19]
        self.vjtPre = int(para[20])
        self.weights = [[] for i in range(LCN)]
        self.weightPos = [[] for i in range(LCN)]
        end = self.bitTrunc
        beg = max(0, self.bitTrunc - 8)
        self.shiftR = beg
        self.truncMask = (1 << (end - beg)) - 1
        self.shiftL = 8 - (end - beg)
        for i in range(LCN):
            for j, w in enumerate(weights[i]):
                if w != 0:
                    self.weights[i].append(w)
                    self.weightPos[i].append(j)

    def truncRelu(self):
        if self.vjtPre <= 0:
            return 0
        return ((int(self.vjtPre) >> self.shiftR) & self.truncMask) << self.shiftL
    def compute(
        self, timeStep, buffer, LCN, SNN_EN, spikeWidth, slotBase, maxPool
    ):
        if not maxPool:
            output = 0
        else:
            output = -(1 << 20)
        for i in range(LCN):
            for w, pos in zip(self.weights[i], self.weightPos[i]):
                if not maxPool:
                    output += w * buffer[i,pos]
                else:
                    output = max(output, buffer[i,pos])
        self.vjtPre += output
        if self.leakPost == 0:
            self.vjtPre += self.leakV
        outputSpike = 0
        if spikeWidth == 0:
            if self.vjtPre >= self.thresholdPos:
                if self.resetMode == 0:
                    self.vjtPre = self.resetV
                elif self.resetMode == 1:
                    self.vjtPre -= self.thresholdPos
                outputSpike = 1
            if self.leakPost == 1:
                self.vjtPre += self.leakV
        else:
            outputSpike = self.truncRelu()
        if SNN_EN == 0:
            self.vjtPre = 0
        spikes = list()
        hardwareSlotNum = Hardware.getAttr("SLOTNUM", True)
        if outputSpike != 0:
            # assert self.tickReltive == 0, f"{self.chipId},{self.coreId},{self.neuronId}"
            for spikeFormat in self.spikeFormats:
                slotId = (self.tickReltive + slotBase) % hardwareSlotNum
                spike = (slotId << 8) + spikeFormat + outputSpike
                spikes.append(spike)

        return spikes

class Core:
    def __init__(self, chipId, coreId, configs):
        self.coreId = coreId
        self.chipId = chipId
        self.weightWidth = 1 << configs['core'][0]
        self.LCN = 1 << configs['core'][1]
        self.inputWidth = configs['core'][2]
        self.spikeWidth = configs['core'][3]
        self.neuronNum = configs['core'][4]
        self.poolMax = configs['core'][5]
        self.tickWaitStart = configs['core'][6]
        self.tickWaitEnd = configs['core'][7]
        self.SNN_EN = configs['core'][8]
        self.targetLCN = 1<<configs['core'][9]
        self.testChipAddr = configs['core'][10]
        self.timeStep = 0
        if self.inputWidth == 0:
            self.inputBuffer = np.zeros([256, 1152])
        else:
            self.inputBuffer = np.zeros([256, 144])
        self.outputBuffer = list()
        self.neurons = list()
        completeNum = len(configs['neuron']) // (self.LCN * self.weightWidth)
        neuronId = 0
        weightBase = 2 ** np.arange(self.weightWidth)
        if self.weightWidth > 1:
            weightBase[self.weightWidth - 1] = -weightBase[self.weightWidth - 1]
        for i in range(completeNum):
            assert 'parameter' in configs['neuron'][neuronId], \
                f"{neuronId} {self.coreId} {self.chipId}"
            para = configs['neuron'][neuronId]['parameter']
            weights = list()
            for j in range(self.LCN):
                weight = 0
                for k in range(self.weightWidth):
                    assert configs['neuron'][neuronId]['weight'] is not None, \
                        f"{self.chipId},{self.coreId},{neuronId}"
                    weight += np.array(configs['neuron'][neuronId]['weight']) * weightBase[k]
                    neuronId += 1
                weights.append(weight)
            self.neurons.append(Neuron(chipId, coreId, i, para, weights,self.weightWidth, self.LCN))

    def receive(self, spike):
        axon = AXONID(spike)
        slot = SLOTID(spike)
        data = DATAID(spike)
        coreId = COREID(spike)
        starId = STARID(spike)
        if ((coreId ^ self.coreId) | starId) == starId:
            self.inputBuffer[slot, axon] = data
    def checkActive(self, timeId):
        return (self.tickWaitStart > 0) and \
            (timeId >= self.tickWaitStart and (self.timeStep < self.tickWaitEnd or self.tickWaitEnd == 0))
    def compute(self, timeId):
        self.outputBuffer.clear()
        if not self.checkActive(timeId):
            return 
        else:
            hardwareSlotNum = Hardware.getAttr("SLOTNUM", True)
            slotBeg = (self.LCN * self.timeStep) % hardwareSlotNum
            slotEnd = (slotBeg + self.LCN)
            slotBase = self.targetLCN * self.timeStep
            for i, neuron in enumerate(self.neurons):
                outputs = neuron.compute(
                    self.timeStep, self.inputBuffer[slotBeg:slotEnd,:],  
                    self.LCN, self.SNN_EN, self.spikeWidth, slotBase, self.poolMax
                )
                if len(outputs) > 0:
                    for output in outputs:
                        self.outputBuffer.append(output)

            self.inputBuffer[slotBeg : slotEnd, :] = 0
    def advanceTime(self, timeId):
        if self.checkActive(timeId):
            self.timeStep+=1

class Chip:
    def __init__(self, chipId):
        self.chipId = chipId
        self.syncTimes = -1
        self.init = 0
        self.cores = dict()
        self.buffer = list()
    def setConfig(self, coreId, config):
        assert coreId not in self.cores
        self.cores[coreId] = Core(self.chipId,coreId, config)
    def setSyncTimes(self, syncTimes):
        self.syncTimes = syncTimes
    def sendSpikes(self):
        offChipSpikes = list()
        for spike in self.buffer:
            if CHIPID(spike) != self.chipId:
                offChipSpikes.append(spike)
            else:
                coreId = COREID(spike)
                self.cores[coreId].receive(spike)
        self.buffer.clear()
        return offChipSpikes
    def receiveSpikes(self, spikes):
        self.buffer.clear()
        for spike in spikes:
            assert CHIPID(spike) == self.chipId
            coreId = COREID(spike)
            starId = STARID(spike)
            cores = {coreId}
            if starId:
                for i in range(10):
                    if (starId >> i) & 1:
                        star = 1 << i
                        coreset = deepcopy(cores)
                        for core in coreset:
                            cores.add(core ^ star)
            for core in cores:
                assert core in self.cores, f"{coreId}, {starId}, {cores} {self.cores.keys()}"
                self.cores[core].receive(spike)

        return 
    def checkActive(self, timeId):
        active = False
        for coreId, core in self.cores.items():
            active = active or core.checkActive(timeId)
        return active
    def compute(self, timeId):
        if timeId > self.syncTimes:
            return
        for coreId, core in self.cores.items():
            core.compute(timeId)
            self.buffer += core.outputBuffer
    def advanceTime(self, timeId):
        for coreId, core in self.cores.items():
            core.advanceTime(timeId)

class Simulator:
    def __init__(self):
        self.chips = dict()
        self.buffer = list()
        self.outputBuffer = list()

    def setConfig(self, configs):
        for coreId, config in configs.items():
            chipId = coreId >> 10
            realCoreId = coreId & MASK.COREMASK
            if chipId not in self.chips:
                self.chips[chipId] = Chip(chipId)
            self.chips[chipId].setConfig(realCoreId, config)

    def setInputs(self, spikes):
        spikesDict = dict()
        for spike in spikes:
            spike = int(spike,2)
            chipId = CHIPID(spike)
            coreId = COREID(spike)
            axonId = AXONID(spike)
            slotId = SLOTID(spike)
            if ISSYNC(spike):
                self.chips[chipId].setSyncTimes(DATAID(spike))
            elif ISINIT(spike):
                continue
            else:
                if chipId not in spikesDict:
                    spikesDict[chipId] = list()
                spikesDict[chipId].append(spike)
        for chipId, s in spikesDict.items():
            assert chipId in self.chips, f"{chipId}:{self.chips.keys()}"
            self.chips[chipId].receiveSpikes(s)

    def advanceTime(self, timeId):
        for chipId, chip in self.chips.items():
            chip.advanceTime(timeId)

    def begin(self):
        active = True
        timeStep = 1
        while active:
            beg = time.time()
            for chipId, chip in self.chips.items():
                chip.compute(timeStep)
                spikes = chip.sendSpikes()
                # for spike in spikes:
                #     if CHIPID(spike) > 3:
                #         assert False, f"{chipId} {CHIPID(spike)} {COREID(spike)} {AXONID(spike)}"
                self.buffer += spikes
            spikesDict = dict()
            for spike in self.buffer:
                chipId = CHIPID(spike)
                if chipId in self.chips:
                    if chipId not in spikesDict:
                        spikesDict[chipId] = list()
                    spikesDict[chipId].append(spike)
                else:
                    self.outputBuffer.append(spike)
            self.buffer.clear()
            for chipId, chip in self.chips.items():
                if chipId in spikesDict:
                    chip.receiveSpikes(spikesDict[chipId])
            self.advanceTime(timeStep)
            timeStep += 1
            active = False
            for chipId, chip in self.chips.items():
                active = active or chip.checkActive(timeStep)
            end = time.time()
            print(f"[{timeStep}]: {end - beg}: outSpike: {len(self.outputBuffer)}")
        return

def parseConfig(configPath):
    
    def parseConfig1(frameGroup):
        pass
    
    def parseConfig2(frameGroup):
        intFrame0 = int(frameGroup[0],2)
        coreId = COREID(intFrame0)
        chipId = CHIPID(intFrame0)
        starId = STARID(intFrame0)
        assert starId == 0
        load = 0
        for i, frame in enumerate(frameGroup):
            payLoad = int(frame,2) & ((1 << 30) - 1)
            load = (load << 30) + payLoad
        load = load >> 23
        config = [0] * 11
        dataLen = [2,4,1,1,13,1,15,15,1,4,10]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        for i in range(10, -1, -1):
            config[i] = load & dataMask[i]
            load >>= dataLen[i]
        return Hardware.getgPlusCoreId2(chipId, coreId, True), config
    
    def parseConfig3(frameGroup):
        intFrame0 = int(frameGroup[0],2)
        coreId = COREID(intFrame0)
        chipId = CHIPID(intFrame0)
        starId = STARID(intFrame0)
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        assert starId == 0
        load = 0
        config = dict()
        dataLen = [8, 11, 5, 5, 5, 5, 5, 5, 2, 30, 1, 5, 1, 29, 29, 1, 1, 30, 1, 5, 30]
        signedData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        memBase = 0

        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 4 == 3:
                tmpConfig = [0] * 21
                for j in range(20, -1, -1):
                    tmpConfig[j] = load & dataMask[j]
                    if signedData[j] and (tmpConfig[j] & (1 << (dataLen[j] - 1)) != 0):
                        tmpConfig[j] -= 1 << dataLen[j]
                    load >>= dataLen[j]
                config[neuronId] = tmpConfig
                neuronId += 1
                memBase = 0
        return Hardware.getgPlusCoreId2(chipId, coreId, True), config
    
    def parseConfig4_param(frameGroup):
        intFrame0 = int(frameGroup[0],2)
        coreId = COREID(intFrame0)
        chipId = CHIPID(intFrame0)
        starId = STARID(intFrame0)
        # neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        neuronId = 0
        assert starId == 0
        memBase = 0
        config = dict()
        dataLen = [8, 11, 5, 5, 5, 5, 5, 5, 2, 30, 1, 5, 1, 29, 29, 1, 1, 30, 1, 5, 30]
        signedData = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
        dataMask = [(1 << (i)) - 1 for i in dataLen]
        load = 0
        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 18 == 17:
                memBase = 0
                for j in range(5):
                    tmpConfig = [0] * 21
                    for i in range(20, -1, -1):
                        tmpConfig[i] = load & dataMask[i]
                        if signedData[i] and (tmpConfig[i] & (1 << (dataLen[i] - 1)) != 0):
                            tmpConfig[i] -= 1 << dataLen[i]
                    load >>= dataLen[i]
                    config[neuronId] = tmpConfig
                    neuronId += 1
                    load >>= 214

        return Hardware.getgPlusCoreId2(chipId, coreId, True), config
    
    def parseConfig4_weight(frameGroup, isSNN):
        intFrame0 = int(frameGroup[0],2)
        coreId = COREID(intFrame0)
        chipId = CHIPID(intFrame0)
        starId = STARID(intFrame0)
        neuronId = (intFrame0 >> 20) & ((1 << 10) - 1)
        if not isSNN:
            neuronId *= 8
        frameNum = intFrame0 & ((1 << 19) - 1)
        assert starId == 0
        memBase = 0
        config = dict()
        load = 0
        for i, frame in enumerate(frameGroup[1:]):
            frame = int(frame,2)
            payLoad = frame & ((1 << 64) - 1)
            load = load + (payLoad << memBase)
            memBase += 64
            if i % 18 == 17:
                memBase = 0
                if isSNN:
                    weight = np.zeros(1152)
                    for i in range(1152):
                        weight[i] = load & 1
                        load >>= 1
                    config[neuronId] = weight
                    neuronId += 1
                else:
                    for j in range(8):
                        weight = np.zeros(144)
                        for i in range(144):
                            weight[i] = load & 1
                            load >>= 1
                        config[neuronId] = weight
                        neuronId += 1
                load = 0
        return Hardware.getgPlusCoreId2(chipId, coreId, True), config

    with open(configPath,'r') as f:
        frames = f.readlines()
    frameNum = len(frames)
    configs =  dict()
    i = 0
    while i < frameNum:
        frame = frames[i].strip()
        intFrame = int(frame,2)
        frameHead = intFrame >> 60
        if frameHead == 0:
            i += 3
            continue
        elif frameHead == 1:
            end = i + 3
            frameGroup = [frames[j].strip() for j in range(i, end)]
            coreId, config = parseConfig2(frameGroup)
            configs[coreId] = {'core':config}
            i = end
        elif frameHead == 2:
            num = intFrame & ((1 << 19) - 1)
            end = i + num + 1
            coreId = COREID(intFrame)
            chipId = CHIPID(intFrame)
            coreId = Hardware.getgPlusCoreId2(chipId, coreId, True)
            neuronNum = configs[coreId]['core'][4]
            frameGroup = [frames[j].strip() for j in range(i, end)]
            coreId, config = parseConfig3(frameGroup)
            isSNN = (configs[coreId]['core'][2] == 0)
            neuronUnit = (1 << configs[coreId]['core'][0]) * (1 << configs[coreId]['core'][1])
            if 'neuron' not in configs[coreId]:
                configs[coreId]['neuron'] = dict()
            if not isSNN:
                for neuronId, neuronConfig in config.items():
                    if neuronId * neuronUnit >= neuronNum:
                        continue
                    
                    for j in range(neuronUnit):
                        if neuronId * neuronUnit + j not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][neuronId * neuronUnit + j] = {'parameter':None, 'weight':None}
                        configs[coreId]['neuron'][neuronId * neuronUnit + j]['parameter'] = neuronConfig
            else:
                for neuronId, neuronConfig in config.items():
                    if neuronId  not in configs[coreId]['neuron']:
                        configs[coreId]['neuron'][neuronId] = {'parameter':None, 'weight':None}
                    configs[coreId]['neuron'][neuronId]['parameter'] = neuronConfig
            i = end
        elif frameHead == 3:
            num = intFrame & ((1 << 19) - 1)
            neuronId = (intFrame >> 20) & ((1 << 10) - 1)
            coreId = COREID(intFrame)
            chipId = CHIPID(intFrame)
            coreId = Hardware.getgPlusCoreId2(chipId, coreId, True)
            end = i + num + 1
            frameGroup = [frames[j].strip() for j in range(i, end)]
            isSNN = (configs[coreId]['core'][2] == 0)
            neuronUnit = configs[coreId]['core'][0] * (1 << configs[coreId]['core'][1])
            neuronNum = configs[coreId]['core'][4]
            if 'neuron' not in configs[coreId]:
                configs[coreId]['neuron'] = dict()
            if isSNN:
                coreId, config = parseConfig4_weight(frameGroup, isSNN)
                for neuronId, neuronConfig in config.items():
                    if neuronId >= neuronNum:
                        continue
                    if neuronId  not in configs[coreId]['neuron']:
                        configs[coreId]['neuron'][neuronId] = {'paramter':None, 'weight': None}
                    configs[coreId]['neuron'][neuronId]['weight'] = neuronConfig
            else:
                if neuronId == 0:
                    coreId, config = parseConfig4_weight(frameGroup, isSNN)
                    for neuronId, neuronConfig in config.items():
                        if neuronId >= neuronNum:
                            continue
                        if neuronId not in configs[coreId]['neuron']:
                            configs[coreId]['neuron'][neuronId] = {'paramter':None, 'weight': None}
                        configs[coreId]['neuron'][neuronId]['weight'] = neuronConfig
                else:
                    coreId, config = parseConfig4_param(frameGroup)
                    neuronUnit = 1<< configs[coreId]['core'][0] #bitWidth
                    neuronUnit *= (1 << configs[coreId]['core'][1]) #LCN
                    neuronBase = int(512 * neuronUnit)
                    for neuronId, neuronConfig in config.items():
                        for i in range(neuronUnit):
                            if neuronBase >= neuronNum:
                                continue
                            if neuronBase not in configs[coreId]['neuron']:
                                configs[coreId]['neuron'][neuronBase] = {'paramter':None, 'weight': None}
                            configs[coreId]['neuron'][neuronBase]['parameter'] = neuronConfig
                            neuronBase += 1
            i = end
        else:
            assert False, frameHead
    return configs

def getData(dataPath):
    with open(dataPath,'r') as f:
        frames = f.readlines()
    return [frame.strip() for frame in frames]

def storeData(frames, outputPath):
    with open(outputPath,'w') as f:
        f.write("\n".join(frames))
        f.write("\n")

def setOnChipNetwork(configPath):
    configs = parseConfig(configPath)
    simulator = Simulator()
    simulator.setConfig(configs)
    return simulator

def runOnChipNetwork(simulator, dataPath, outputPath):
    dataFrames = getData(dataPath)
    simulator.setInputs(dataFrames)
    simulator.begin()
    frames = ["{:064b}".format(frame) for frame in simulator.outputBuffer]
    storeData(frames, outputPath)
    return

def loadOutputFormat(baseDir):
    infoDir = os.path.join(baseDir, "info")
    formatDir = os.path.join(baseDir, "formats")
    with open(os.path.join(infoDir, "outDict.json"),"r") as f:
        outDict = json.load(f)
    with open(os.path.join(infoDir, "shape.json"),"r") as f:
        shapeDict = json.load(f)
    with open(os.path.join(infoDir, "scale.json"),"r") as f:
        scaleDict = json.load(f)
    with open(os.path.join(formatDir, "mapper.txt"),"r") as f:
        mapper = json.load(f)
    intMapper = dict()
    for name, [tensorName, pos] in mapper.items():
        intMapper[int(name)] = [tensorName, int(pos)]
    return outDict, shapeDict, scaleDict, intMapper

def check(fullNet, dataDict, scaleDict, *data):
    if fullNet is None:
        return
    fullNet(*data)
    for name, outData in dataDict.items():
        if hasattr(fullNet.nodes[name], "timesteps"):
            gt = fullNet.nodes[name].to_float()
            # gt = fullNet.nodes[name].data
        else:
            gt = fullNet.nodes[name].detach() * np.array(scaleDict[name])
        print(name)
        print("---- GroundTruth result shape & sum ---------")
        print(gt.shape)
        print(gt.sum())
        print("----- onCHIP   result shape & sum ---------")
        print(outData.shape)
        print(outData.sum())
        assert (gt == outData).int().sum() == np.prod(list(outData.size()))
    print("check passed.\n")

def runFullApp(preNet, postNet, baseDir, data, fullNet, timeStep):
    frameFormats, frameNums, inputNames = loadInputFormats(baseDir)
    dataDict = runPreNet(preNet, inputNames, *data)
    
    inputPath = os.path.join(baseDir, "frames/input.txt")
    encodeDataFrame(dataDict, frameFormats, frameNums, inputNames, inputPath)

    configPath = os.path.join(baseDir, "frames/config.txt")
    simulator = setOnChipNetwork(configPath)
    
    outputPath = os.path.join(baseDir, "frames/simuOut.txt")
    runOnChipNetwork(simulator, inputPath, outputPath)
    
    dataFrames = getData(outputPath)
    outDict, shapeDict, scaleDict, mapper = loadOutputFormat(baseDir)
    newDataDict = decodeDataFrame(dataFrames, outDict, shapeDict, scaleDict, mapper, timeStep)
    # outdataDict = runPostNet(postNet, newDataDict)
    check(fullNet, newDataDict, scaleDict, *data)
    return

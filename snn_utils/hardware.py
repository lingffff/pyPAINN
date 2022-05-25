from copy import deepcopy
class Hardware:

    OUTPUTBEG = (1 << (0 + 5 + 5 + 5 + 17))
    # OUTPUTPOS = (1 << (3 + CHIPYBIT + COREXBIT + COREYBIT + COREBASE))

    @staticmethod
    def getCoreId(fullId, offline):
        if offline:
            return HardwareF.getCoreId(fullId)
        else:
            return HardwareN.getCoreId(fullId)
    
    @staticmethod
    def getGroupId(fullId, offline):
        if offline:
            return HardwareF.getGroupId(fullId)
        else:
            return HardwareN.getGroupId(fullId)

    @staticmethod
    def addGroupId(groupId, cPlusUnitId, offline):
        if offline:
            return HardwareF.addGroupId(groupId, cPlusUnitId)
        else:
            return HardwareN.addGroupId(groupId, cPlusUnitId)

    @staticmethod
    def getgPlusCoreId(fullId, offline):
        if offline:
            return HardwareF.getgPlusCoreId(fullId)
        else:
            return HardwareN.getgPlusCoreId(fullId)

    @staticmethod
    def getgPlusCoreId2(groupId, coreId, offline):
        if offline:
            return HardwareF.getgPlusCoreId2(groupId, coreId)
        else:
            return HardwareN.getgPlusCoreId2(groupId, coreId)
    
    @staticmethod
    def getNeuronId(fullId, offline):
        if offline:
            return HardwareF.getNeuronId(fullId)
        else:
            return HardwareN.getNeuronId(fullId)

    @staticmethod
    def getAxonId(fullId, inputWidth, offline):
        if offline:
            return HardwareF.getAxonId(fullId, inputWidth)
        else:
            return HardwareN.getAxonId(fullId)
    
    @staticmethod
    def getComAxonId(fullId, offline):
        if offline:
            return HardwareF.getComAxonId(fullId)
        else:
            return HardwareN.getComAxonId(fullId)

    @staticmethod
    def getSlotId(fullId, inputWidth, offline):
        if offline:
            return HardwareF.getSlotId(fullId, inputWidth)
        else:
            return HardwareN.getSlotId(fullId)
    
    @staticmethod
    def getfullId(groupId, coreId, unitId, offline):
        if offline:
            return HardwareF.getfullId(groupId, coreId, unitId)
        else:
            return HardwareN.getfullId(groupId, coreId, unitId)

    @staticmethod
    def getfullId2(gPlusCoreId, unitId, offline):
        if offline:
            return HardwareF.getfullId2(gPlusCoreId, unitId)
        else:
            return HardwareN.getfullId2(gPlusCoreId, unitId)
    
    @staticmethod
    def addBaseCoreId(fullId, offline):
        if offline:
            return HardwareF.addBaseCoreId(fullId)
        else:
            return HardwareN.addBaseCoreId(fullId)

    @staticmethod
    def getAttr(name, offline):
        if offline:
            return getattr(HardwareF, name)
        else:
            return getattr(HardwareN, name)

    @staticmethod
    def setNoCLevel(NoCLevelsX, NoCLevelsY, offline):
        if offline:
            HardwareF.setNoCLevel(NoCLevelsX, NoCLevelsY)
        else:
            HardwareN.setNoCLevel(NoCLevelsX, NoCLevelsY)

#NeuronId: ---|------------------|---------|------------------------|
#               COREX + COREY(10)  unuse(5)      NEURON(12)
#AxonId  : ---|------------------|---- ------|----------------------|
#               COREX + COREY(10)  SLOT(6)       AXONBIT(11)
class HardwareF:

    COREXBIT  = 5
    COREYBIT  = 5
    CHIPXBIT  = 5
    CHIPYBIT  = 5
    SLOTBIT   = 6
    AXONBIT   = 11
    NEURONBIT = 12
    AXONSLOT = (AXONBIT + SLOTBIT)
    # UNITBIT = max(AXONBIT, NEURONBIT)
    # COREBASE = (UNITBIT + SLOTBIT)
    COREBASE = max(AXONSLOT, NEURONBIT)
    COREMASK = ((1 << (COREXBIT + COREYBIT)) - 1)
    GROUPBASE = (COREBASE + COREXBIT + COREYBIT)

    COREXNUM  = 32
    COREYNUM  = 32
    CHIPXNUM  = 32
    CHIPYNUM  = 32
    SLOTNUM  = 256
    NOCLEVEL = 5
    AXONNUM   = 1152
    NEURONNUM = 512
    MAXLCN = 64

    NoCLevelsX = [2 for i in range(NOCLEVEL)]
    NoCLevelsY = [2 for i in range(NOCLEVEL)]

    @staticmethod
    def setNoCLevel(NoCLevelsX, NoCLevelsY):
        HardwareF.NoCLevelsX = deepcopy(NoCLevelsX)
        HardwareF.NoCLevelsY = deepcopy(NoCLevelsY)
        HardwareF.NOCLEVEL = len(NoCLevelsX)


    @staticmethod
    def getCoreId(fullId):
        return (fullId >> HardwareF.COREBASE) & HardwareF.COREMASK
    
    @staticmethod
    def getGroupId(fullId):
        return (fullId >> HardwareF.GROUPBASE)

    @staticmethod
    def addGroupId(groupId, cPlusUnitId):
        return (groupId << HardwareF.GROUPBASE) + cPlusUnitId

    @staticmethod
    def getgPlusCoreId(fullId):
        return (fullId >> HardwareF.COREBASE)

    @staticmethod
    def getgPlusCoreId2(groupId, coreId):
        return ((groupId) << (HardwareF.COREXBIT + HardwareF.COREYBIT)) + coreId

    @staticmethod
    def getNeuronId(fullId):
        return (fullId & ((1<<(HardwareF.NEURONBIT)) - 1))

    @staticmethod
    def getAxonId(fullId, inputWidth):
        return HardwareF.getComAxonId(fullId) % (HardwareF.AXONNUM // inputWidth)
    
    @staticmethod
    def getComAxonId(fullId):
        return fullId & ((1 << HardwareF.COREBASE) - 1)

    @staticmethod
    def getSlotId(fullId, inputWidth):
        return HardwareF.getComAxonId(fullId) // (HardwareF.AXONNUM // inputWidth)
    
    @staticmethod
    def getfullId(groupId, coreId, unitId):
        return (groupId << HardwareF.GROUPBASE) + (coreId << HardwareF.COREBASE) + unitId

    @staticmethod
    def getfullId2(gPlusCoreId, unitId):
        return (gPlusCoreId << HardwareF.COREBASE) + unitId
    
    @staticmethod
    def addBaseCoreId(fullId):
        return fullId

# NeuronId: ---|------------------|--------|------------|
#               COREX + COREY(10)   SLOT(3)  NEURON(10)
class HardwareN: # online hardware configuration
    COREXBEG = 14
    COREYBEG = 14
    
    COREXNUM = 4
    COREYNUM = 4
    SLOTNUM = 8
    NOCLEVEL = 2
    NEURONUM = 1024
    AXONNUM = 1024

    AXONBIT = 10
    NEURONBIT = 10
    SLOTBIT = 3
    COREBIT = 10
    CHIPBIT = 10

    COREBASE = (AXONBIT + SLOTBIT)
    GROUPBASE = (COREBASE + COREBIT)
    
    COREMASK = (1 << COREBIT) - 1
    CHIPMASK = (1 << CHIPBIT) - 1
    NEURONMASK = (1 << NEURONBIT) - 1
    AXONMASK = (1 << AXONBIT) - 1
    COMAXONMASK = (1 << (SLOTBIT + AXONBIT)) - 1
    COREBASEMASK = ((1 << (5 - 2)) - 1) + (((1 << (5 - 2)) - 1) << 5)

    NoCLevelsX = [2 for i in range(NOCLEVEL)]
    NoCLevelsY = [2 for i in range(NOCLEVEL)]

    @staticmethod
    def setNoCLevel(NoCLevelsX, NoCLevelsY):
        HardwareN.NoCLevelsX = deepcopy(NoCLevelsX)
        HardwareN.NoCLevelsY = deepcopy(NoCLevelsY)
        HardwareN.NOCLEVEL = len(NoCLevelsX)

    @staticmethod
    def getGroupId(fullId):
        return fullId >> HardwareN.GROUPBASE
    
    @staticmethod
    def getCoreId(fullId):
        return (fullId >> HardwareN.COREBASE) &  HardwareN.COREMASK
    
    @staticmethod
    def getNeuronId(fullId):
        return fullId & HardwareN.NEURONMASK
    
    @staticmethod
    def getAxonId(fullId):
        return fullId & HardwareN.AXONMASK
    
    @staticmethod
    def getComAxonId(fullId):
        return fullId & HardwareN.COMAXONMASK

    @staticmethod
    def addGroupId(groupId, cPlusUnitId):
        return (groupId << HardwareN.GROUPBASE) + cPlusUnitId

    @staticmethod
    def getgPlusCoreId(fullId):
        return (fullId >> HardwareN.COREBASE)

    @staticmethod
    def getgPlusCoreId2(groupId, coreId):
        return ((groupId) << (HardwareN.COREBIT)) + coreId

    @staticmethod
    def getAxonId(fullId):
        return HardwareN.getComAxonId(fullId) % HardwareN.AXONNUM
    
    @staticmethod
    def getComAxonId(fullId):
        return fullId & HardwareN.COMAXONMASK

    @staticmethod
    def getSlotId(fullId):
        return HardwareN.getComAxonId(fullId) // HardwareN.AXONNUM
    
    @staticmethod
    def getfullId(groupId, coreId, unitId):
        return (groupId << HardwareN.GROUPBASE) + (coreId << HardwareN.COREBASE) + unitId

    @staticmethod
    def getfullId2(gPlusCoreId, unitId):
        return (gPlusCoreId << HardwareN.COREBASE) + unitId

        
    @staticmethod
    def addBaseCoreId(fullId):
        assert (fullId & HardwareN.COREBASEMASK) == 0
        return fullId + HardwareN.COREBASEMASK

# data frame
# |------|---------|---------|---------|--|--------|-----------|----------|
#  0x0200  CHIP(10)  CORE(10)  STAR(10)  3  Axon(11)  Slot(8/3)   Data(8)
class DataFrame:
    MASK1 = (1 << 20) - 1
    MASK2 = (1 << 11) - 1
    MASK3 = (1 << 10) - 1
    MASK4 = (1 << 8) - 1
    MASK5 = (1 << 30) - 1
    @staticmethod
    def getCoreId(fullId):
        return (fullId >> 40) & DataFrame.MASK1
    @staticmethod
    def getStarId(fullId):
        return (fullId >> 30) & DataFrame.MASK3
    @staticmethod
    def genFakeFrame(coreId, starId, axonId, slotId):
        assert coreId <= DataFrame.MASK1, coreId
        return (coreId << 40) + (starId << 30) + (axonId << 16) + (slotId << 8)
    @staticmethod
    def getPayLoad(fullId):
        return fullId & DataFrame.MASK5
    
    @staticmethod
    def getFrame(oldId, coreAddr, starAddr):
        return (8 << 60) + (coreAddr << 40) + (starAddr << 30) + oldId
    
    @staticmethod
    def getFormat(oldId, coreAddr, starAddr):
        return DataFrame.getFrame(oldId, coreAddr, starAddr) >> 8


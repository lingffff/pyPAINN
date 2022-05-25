import os
import torch


def loadNet(netPath):
    net = torch.load(netPath,map_location="cpu")
    net = net.cpu()
    net.eval()
    return net

def loadAuxNet(auxNetDir):
    preNetPath = os.path.join(auxNetDir, "preNet.pth")
    postNetPath = os.path.join(auxNetDir, "postNet.pth")
    return loadNet(preNetPath), loadNet(postNetPath)
# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: PAICore2.0-FPGA-Ethernet cifar10 classification demo.
# Create: 2021-5-23
from calendar import c
import torchvision.transforms as transforms
import cv2
import torch
from snn_utils.PAI_board import PAIBoard
import time

if __name__ == '__main__':
    # input
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2470, 0.2435, 0.2616])
    transform_cifar = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((32, 32)),
                normalize,
            ])
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # prepare: baseDir is from SNNToolChain
    snn = PAIBoard(baseDir="./es", timestep=64)
    snn.connect()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        img_t = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform_cifar(img_t)
        # inference
        out = snn(img_tensor)
        probs = torch.softmax(out, dim=0)
        pred = probs.argmax().item()

        cv2.putText(img, '%s: %.4f' % (classes[pred], probs[pred].item()), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow('Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
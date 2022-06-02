# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: PAICore2.0-FPGA-Ethernet application example.
# Create: 2021-5-12
import torchvision.transforms as transforms
import cv2

from snn_utils.PAI_board import PAIBoard
import time

if __name__ == '__main__':
    # input
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2470, 0.2435, 0.2616])
    transform_cifar = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # prepare: baseDir is from SNNToolChain
    snn = PAIBoard(baseDir="./es", timestep=64)
    snn.connect()

    loop = 100
    t0 = time.time()
    for i in range(loop):
        print(f"\n# Test {i + 1}")
        # input image
        img = cv2.imread("./files/cat.png")
        img_tensor = transform_cifar(img)
        print(img_tensor.size())
        # inference
        out = snn(img_tensor)
        pred = out.argmax().item()
        print(out, f"=> {classes[pred]}")
    t1 = time.time()
    avg_time = (t1 - t0) / loop
    print(f"\n### Avg time: {avg_time}s")

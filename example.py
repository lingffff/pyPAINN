# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: PAICore2.0-FPGA-Ethernet application example.
# Create: 2021-5-12
import torchvision.transforms as transforms
import cv2

from snn_utils.PAI_board import PAIBoard


if __name__ == '__main__':
    # input
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2470, 0.2435, 0.2616])
    transform_cifar = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    loop = 100
    for i in range(loop):
        print(f"\n==> Test {i + 1}")
        img = cv2.imread("./files/cat.png")
        img_tensor = transform_cifar(img)
        print(img_tensor.size())
        # prepare: baseDir is from SNNToolChain
        snn = PAIBoard(baseDir="./es", timestep=64)
        snn.connect()
        # inference
        out = snn(img_tensor)
        print(out)

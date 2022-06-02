import os
from snn_utils.Ethernet_utils.frameHandler import FrameHandler

def run_cmd(task_name, cmd):
    print(f"=== Run {task_name}, CMD: === \n=== {cmd} ===")
    code = os.system(cmd)
    if code != 0:
        print(f"{task_name} Error")
        exit()

def sendFrame(name: str):
    eth = FrameHandler()
    eth.ethernetHandler.reconnect()
    eth.transformFrameFile(name) 
    eth.writeWithGapFrameNum(name, 1000, 1)


if __name__ == '__main__':
    baseDir = "./es"
    configFramePath = os.path.join(baseDir, "frames/config.txt")
    run_cmd("Move config frames", f"cp {configFramePath} ./files/")
    sendFrame("config")
    print("Config done.")

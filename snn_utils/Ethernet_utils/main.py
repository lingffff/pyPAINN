import sys
import os
import logging
import frameHandler
import mylog
import threading
import time

#设置log文件的格式
logging.basicConfig(level=logging.DEBUG,
                    filename='log.log',
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')

frameHandler = frameHandler.FrameHandler()      #处理帧文件的类

#先启动从ethernet设备读数据的线程，保证在程序结束前不停的读出。
readingThread = threading.Thread(target=frameHandler.read, name='ReadingThread')
#readingThread.start()
        
        
#下面执行写入过程
inputData = None
fileName = 'None'
transform = None
while True:
    #输入需要执行的操作命令
    inputData = input()

    if inputData == 'exit':     #退出程序
        frameHandler.stopReading()
        time.sleep(1)
        frameHandler.ethernetHandler.close()
        sys.exit(0)
    elif inputData == 'ethernet':    #重新查找、连接ethernet设备
        frameHandler.ethernetHandler.reconnect()
        continue
    elif inputData == 'read':   #启动设备读取
        frameHandler.stopReading()
        time.sleep(0.005)
        frameHandler.startReading()
        readingThread = threading.Thread(target=frameHandler.read, name='ReadingThread')
        readingThread.start()
        print('读出线程已重启。')
        continue
    elif inputData == 'stopread':
        frameHandler.stopReading()
        print('读出线程已关闭。')
        continue
    elif inputData == 'file':   #输入帧文件的信息，输入格式为'文件名(不含后缀),是否为新文件'
        print('当前帧文件：%s，请输入“文件名(不含后缀),是否为新文件(1或0)”：' % fileName)
        inputDataB = input()
        try:
            fileName, transform = inputDataB.split(',')
            #检查帧文件是否存在，如果不存在，跳出循环一次
            if not os.path.exists('./files/%s.txt' % fileName):
                mylog.error('Error: \'%s.txt\' does not exit.' % (fileName))
                continue
            if transform == '1':
                frameHandler.transformFrameFile(fileName)
            if transform == '0':
                print('帧文件转换完成。请输入指令auto、hand、gap、handgap：')
        except BaseException as e:
            mylog.error("Error: 指令不符合要求。")
            mylog.error(str(e))
            continue
    elif inputData == 'auto': #自动将整个文件的数据帧写入
        print('开始自动写入帧文件：%s...' % fileName)
        if frameHandler.write(fileName):
            print('自动写入结束。')
        else:
            print('自动写入失败。')
    elif inputData == 'hand': #写入指定数目的数据帧
        print('开始手动写入帧文件：%s，请输入数字或stop：' % fileName)
        if frameHandler.writeWithFrameNum(fileName):
            print('已退出手动写入。')
        else:
            print('手动写入失败。')
    elif inputData == 'gap': #写入特定数据帧的间隔时间
        print('开始分块写入帧文件：%s，请输入间隔时间，单位ms' % fileName)
        if frameHandler.writeWithGap(fileName):
            print('分块写入结束。')
        else:
            print('手动写入失败。')
    elif inputData == 'handgap': #写入分块数据帧的数目和间隔时间
        print('开始分块写入帧文件：%s，请输入“分块数目,间隔时间”，单位ms' % fileName)
        inputDataB = input()
        try:
            number, gap = inputDataB.split(',')
            if frameHandler.writeWithGapFrameNum(fileName, number, gap):
                print('分块写入结束。')
            else:
                print('手动写入失败。')
        except BaseException as e:
            mylog.error("Error: 指令不符合要求。")
            mylog.error(str(e))
            continue
    else:
        print('指令未识别，请重新输入：')
        continue
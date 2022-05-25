# encoding: utf-8
import numpy
import time
import array
import ethernetHandler
import logging
import mylog

class FrameHandler():

    def __init__(self):
        self.ethernetHandler = ethernetHandler.EthernetHandler()
        self.allowRead = True

    def transfer(self,frameData):
		#把40bit的数据转成64比特的帧数据
        return frameData[56:64]+frameData[48:56]+frameData[40:48]+frameData[32:40]+frameData[24:32]+frameData[16:24]+frameData[8:16]+frameData[0:8]+'\n'

    def transformFrameFile(self,fileName):
        r'''
        先把64bit的TXT帧文件转成64bit的TXT帧文件，再转成bytes类型的.npy文件，用于传输；
        '''
        #读入64比特的帧数据
        mylog.info('正在将64bit帧文件转换为64bit的倒序帧文件...')
        with open('./files/%s.txt' % fileName,'r') as net40b:
            frames = net40b.readlines()
            framesTrans = [self.transfer(f) for f in frames]

        #写入64比特的帧数据,fileName64b_reverse.txt 
        with open('./files/%s_64b_reverse.txt' % fileName,'w') as net64b:
            temp=[net64b.write(f) for f in framesTrans]
            mylog.info('正在将64bit的帧文件转换为.npy文件...')

        #读入按字节倒序的64比特的帧数据，并转成bytes类型数据，存入.npy文件中
        frames = framesTrans
        framesNum = len(frames)
        tempData = numpy.zeros(framesNum).astype('uint64')
        for i,f in enumerate(frames):
            tempData[i] = int(f,2)

        tempData.dtype = 'uint8'
        frameBytes = bytes(tempData)
        numpy.save('./files/%s.npy' % fileName,numpy.array(frameBytes))
        mylog.info('帧文件转换完成。请输入指令auto、hand、gap、handgap：')

    def write(self,fileName):
        r'''
        把处理好的帧数据连续写入ethernet设备；
        '''
        try:
            mylog.info('正在写入帧文件...')
            frameData = numpy.load('./files/%s.npy' % fileName)
            #某些情况下需要降低发送速度，下面将帧数据分块发送；
            frameData = bytes(frameData)
            dataLength = len(frameData)
            bytesValidOneTime = int(1024)  #每次发送的数据中，有效帧的字节数
            i = 0
            while (i+bytesValidOneTime) < dataLength:
                if not self.ethernetHandler.write(frameData[i:(i+bytesValidOneTime)]):
                    mylog.error('Error: 写入失败！')
                    return False
                i += bytesValidOneTime
            if i < dataLength:
                if not self.ethernetHandler.write(frameData[i:]):
                    mylog.error('Error: 写入失败！')
                    return False
        except BaseException as e:
            mylog.error("Error: 写入帧文件失败。")
            mylog.error(str(e))
            return False
        else:
            mylog.info('已成功写入帧文件。')
            return True

    def writeWithGap(self,fileName):
        r'''
        把处理好的帧数据逐帧写入ethernet设备，每帧间隔gap，单位为us；
        '''
        #先定义同步帧和测试帧
        syncFrame = 9      #1001
        clearFrame = 10    #1010
        initialFrame = 11  #1011
        testFrame1 = 4     #0100
        testFrame2 = 5     #0101
        testFrame3 = 6     #0110
        testFrame4 = 7     #0111
        try:
            try:
                gap = int(input())
            except BaseException as e:
                mylog.error("Error: 指令不符合要求，请输入数字")
                mylog.error(str(e))
            mylog.info('正在写入帧文件...')
            frameData = numpy.load('./files/%s.npy' % fileName)
            #某些情况下需要降低发送速度，下面将帧数据分块发送；
            frameData = bytes(frameData)
            dataLength = len(frameData)
            i = 0
            bytesValidOneTime = int(1024)  #每次发送的数据中，有效帧的字节数
            while i < dataLength:
                time.sleep(gap/1000) #ms
                j = i
                while j < dataLength:
                    frameTitle = frameData[j] >> 4
                    if frameTitle == syncFrame or frameTitle == clearFrame or frameTitle == initialFrame or frameTitle == testFrame1 or frameTitle == testFrame2 or frameTitle == testFrame3 or frameTitle == testFrame4:
                        break
                    j += 8
                while (i+bytesValidOneTime) < j+8:
                    if not self.ethernetHandler.write(frameData[i:(i+bytesValidOneTime)]):
                        mylog.error('Error: 写入失败！')
                        return False
                    i += bytesValidOneTime
                if i < j+8:
                    if not self.ethernetHandler.write(frameData[i:j+8]):
                        mylog.error('Error: 写入失败！')
                        return False
                i = j+8
        except BaseException as e:
            mylog.error("Error: 写入帧文件失败。")
            mylog.error(str(e))
            return False
        else:
            mylog.info('已成功写入帧文件。')
            return True
            
    def writeWithFrameNum(self,fileName):
        r'''
        把固定数量的帧数据写入ethernet设备，起始帧编号为currentFrameNo，写入帧的数量为frameNum；
        '''
        currentFrameNo = 0
        try:
            frameData = numpy.load('./files/%s.npy' % fileName)
            frameData = bytes(frameData)
            dataLength = len(frameData)
            frameLength = dataLength//8
        except BaseException as e:
            mylog.error("Error: 读取帧文件失败。")
            mylog.error(str(e))
            return False
        else:
            mylog.info('已成功读取帧文件，当前帧数目为：%d' % frameLength)
        while True:
            inputData = input()
            if inputData == 'stop':
                return True
            else:
                try:
                    frameNum = int(inputData)
                except BaseException as e:
                    mylog.error("Error: 指令不符合要求，请输入数字或stop")
                    mylog.error(str(e))
                    continue
                if currentFrameNo+frameNum > frameLength:
                    mylog.error('Error: 超出帧文件范围。起始帧为%d，写入帧数量为%d...' % (currentFrameNo,frameNum))
                    return False
                mylog.info('正在写入帧，起始帧为%d，帧数量为%d...' % (currentFrameNo,frameNum))
                framesValidOneTime = int(128)  #每次发送的数据中，有效帧的数目
                i = 0
                while (i+framesValidOneTime) < frameNum:
                    #time.sleep(0.001)
                    if not self.ethernetHandler.write(frameData[(8*(currentFrameNo+i)):(8*(currentFrameNo+i+framesValidOneTime))]):
                        mylog.error('Error: 写入失败！')
                        return False
                    i += framesValidOneTime
                if i < frameNum:
                    if not self.ethernetHandler.write(frameData[(8*(currentFrameNo+i)):(8*(currentFrameNo+frameNum))]):
                        mylog.error('Error: 写入失败！')
                        return False	
                currentFrameNo += frameNum
                print('已写入，当前还剩%d帧，请输入数字或stop：' % (frameLength-currentFrameNo))
        return True
    
    def writeWithGapFrameNum(self,fileName,number,gap):
        r'''
        把固定数量的帧数据间隔一段时间写入ethernet设备，每次写入帧的数量为frameNum；
        '''
        currentFrameNo = 0
        try:
            frameData = numpy.load('./files/%s.npy' % fileName)
            frameData = bytes(frameData)
            dataLength = len(frameData)
            frameLength = dataLength//8
        except BaseException as e:
            mylog.error("Error: 读取帧文件失败。")
            mylog.error(str(e))
            return False
        else:
            mylog.info('已成功读取帧文件，当前帧数目为：%d' % frameLength)
        while True:
            try:
                frameNum = int(number)
                gap = int(gap)
            except BaseException as e:
                mylog.error("Error: 指令不符合要求")
                mylog.error(str(e))
                continue
            mylog.info('正在写入帧，每次写入帧数量为%d...' % (frameNum))
            framesValidOneTime = int(128)  #每次发送的数据中，有效帧的数目
            while currentFrameNo+frameNum < frameLength:
                i = 0
                while (i+framesValidOneTime) < frameNum:
                    #time.sleep(0.001)
                    if not self.ethernetHandler.write(frameData[(8*(currentFrameNo+i)):(8*(currentFrameNo+i+framesValidOneTime))]):
                        mylog.error('Error: 写入失败！')
                        return False
                    i += framesValidOneTime
                if i < frameNum:
                    if not self.ethernetHandler.write(frameData[(8*(currentFrameNo+i)):(8*(currentFrameNo+frameNum))]):
                        mylog.error('Error: 写入失败！')
                        return False	
                currentFrameNo += frameNum
                time.sleep(gap/1000) #ms
            frameNum = frameLength - currentFrameNo
            if frameNum > 0:
                i = 0
                while (i+framesValidOneTime) < frameNum:
                    #time.sleep(0.001)
                    if not self.ethernetHandler.write(frameData[(8*(currentFrameNo+i)):(8*(currentFrameNo+i+framesValidOneTime))]):
                        mylog.error('Error: 写入失败！')
                        return False
                    i += framesValidOneTime
                if i < frameNum:
                    if not self.ethernetHandler.write(frameData[(8*(currentFrameNo+i)):(8*(currentFrameNo+frameNum))]):
                        mylog.error('Error: 写入失败！')
                        return False	
                currentFrameNo += frameNum
                time.sleep(gap/1000) #ms
            print('已写入%d帧' % (currentFrameNo))
            return True
        return True

    def read(self):
        while self.allowRead:
            readOutBytes = self.ethernetHandler.read()
            with open('./files/out.txt','a') as outFile:
                readOutBytesNum = len(readOutBytes)
                for i in range(0,readOutBytesNum,8):
                    for j in range(8):
                        outFile.write('{:0>8b}'.format(readOutBytes[i+j]))
                    outFile.write('\n')
    r'''
    for j in range(3,-1,-1):
	    outFile.write('{:0>8b}'.format(readOutBytes[i+j]))
    for j in range(7,3,-1):
	    outFile.write('{:0>8b}'.format(readOutBytes[i+j]))
    outFile.write('\n')
    '''

    def stopReading(self):
        self.allowRead = False

    def startReading(self):
        self.allowRead = True
# -*- coding: utf-8 -*-
import threading
import time
import numpy as np

class Pro(object):
    def __init__(self):
    # threading.Thread.__init__(self)
        self._sName = "machao"

    def echo(self):
        while True:
            print("start")
            time.sleep(3)


class Test(object):
  def __init__(self):
    # threading.Thread.__init__(self)
    self._sName = "machao"
    self.pro = Pro()
 
  def process(self):
    #args是关键字参数，需要加上名字，写成args=(self,)
    th1 = threading.Thread(target=self.pro.echo)
    th1.start()
    # th1.join()
 
 
s = np.array(["000", "001", "010", "011"])

index = [0, [1,2]]
print(s[index])
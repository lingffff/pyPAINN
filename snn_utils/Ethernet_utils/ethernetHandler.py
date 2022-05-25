import socket
import mylog

r'''
ethernetHandler用于处理与ethernet接口有关的任务
'''

class EthernetHandler():

    def __init__(self):
        r'''
        连接ethernet设备。
        '''
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sendaddr = ('192.168.0.2',8080)
        self.localaddr = ('192.168.0.3',8080)
        self.udp_socket.bind(self.localaddr)
        mylog.info("ethernet已连接")
        

    def write(self,data):
        r"""Write data to ethernet.
        The data parameter should be a sequence like type convertible to
        the array type (see array module).
        """
        return self.udp_socket.sendto(data,self.sendaddr)

    def read(self):
        r"""Read data from ethernet.
        The bulksNum parameter should be the amount of bulks to be readed.
        One bulk is wMaxPacketSize(512) bytes.
        The method returns the data readed as an array. If read nothing, return None.
        """
        data = self.udp_socket.recvfrom(1024)
        return data[0]
        
    def close(self):
        self.udp_socket.close()
        
    def reconnect(self):
        self.udp_socket.close()
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(self.localaddr)
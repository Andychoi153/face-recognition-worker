import socket
import sys
import os

from time import sleep


HOST = ""
PORT = 9999


def inloop_Match(socket, name):
    sleep(1)

def inloop_recieveFile(socket, addr):
    #Send Ready Signal
    socket.sendall('Ready'.encode())

    #Receive filename
    msg = socket.recv(4096)
    filename =msg.decode('utf-8')
    print(filename)

    #Open file
    f = open(str(addr) + filename, "wb")

    #Receive File Data
    while True:
        # get file bytes
        data = socket.recv(4096)
        if not data:
            break
        # write bytes on file
        f.write(data)
    print(filename, " download complete")
    f.close()


#Main

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(5)

while True:
    try:
        requester_socket, requester_addr = s.accept()
        inloop_recieveFile(requester_socket, requester_addr)
        inloop_Match(requester_socket, 'socket_a')
    except Exception as e:
        print(str(e))
        s.close()
        print("Disconnected")
        sys.exit()

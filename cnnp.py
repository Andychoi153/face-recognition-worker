import cv2
import usb.core
import usb.util
import array
import numpy as np
from numpy import genfromtxt

# Receive one frame and do nothing
def getFrame():
    epIn = 0x81
    cnn_array = array.array('B')

    data = dev.read(epIn, 16384, 2000)

    if len(data) != 2:
        return None

    #cnn_array.extend(data)
    read_size = 0

    data = dev.read(epIn, 16384, 200)

    while len(data) != 2:
        read_size = read_size + len(data)
        cnn_array.extend(data)
        data = dev.read(epIn, 16384, 200)

def sendDetectedFace(faceROI):
    epOut = 0x1
    outbuf = array.array('B')
    outbuf.fromlist(np.reshape(faceROI, 16384).tolist())

    write_len = dev.write(epOut, outbuf, 200)

    return write_len

def sendNullFace():
    epOut = 0x1
    outbuf = array.array('B')
    outbuf.fromlist([0]*16384)

    write_len = dev.write(epOut, outbuf, 200)

    return write_len

def getCNNOut():
    epIn = 0x81
    cnn_array = array.array('B')

    data = dev.read(epIn, 16384, 2000) # CNN Processing Delay

    if len(data) != 2:
        return None

    #cnn_array.extend(data)
    read_size = 0

    data = dev.read(epIn, 16384, 200)

    while len(data) != 2:
        read_size = read_size + len(data)
        cnn_array.extend(data)
        data = dev.read(epIn, 16384, 200)

    #print(read_size)
    #return cnn_array

    if read_size < 9600:
        return None

    cnnOut = []
    for i in range(0, 4800):
        temp = cnn_array[i*2] + cnn_array[i*2+1]*256

        if temp > 32767:
            temp = temp - 65536

        cnnOut.append(temp/256)

    #cnnOut2 = np.array(cnnOut)

    cnnResult = np.reshape(np.matmul(np.reshape(np.array(cnnOut), (1, 4800)), getCNNOut.fc1_weight), (512))
    cnnResultNorm = cnnResult/np.linalg.norm(cnnResult)

    return cnnResultNorm
getCNNOut.fc1_weight = genfromtxt('FaceRecognitionWorker/config/fc1_weight.csv', delimiter=',')

def cnnpRun(faceROI):
    ## Receive one frame and do nothing
    getFrame()

    ## Original code
    sendDetectedFace(faceROI)
    return getCNNOut()

dev = usb.core.find(idVendor=0x04b4, idProduct=0x00F1)
                


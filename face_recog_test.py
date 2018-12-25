import cv2
import numpy as np
import requests
import json
import pandas as pd
import socket
import multiprocessing
import time
import csv
import argparse
from FaceRecognitionWorker.log import log
from PIL import Image

import io
from cnnp import cnnpRun

HOST = ""
PORT = 9999


class FaceRecognitionWorker:

    def __init__(self):
        ##
        self.faceCascade = cv2.CascadeClassifier('FaceRecognitionWorker/config/haarcascade_frontalface_alt2.xml')

        ##
        # TODO: DB 형식의 csv 파일 포맷 생성
        # self.db = pd.read_csv('FaceRecognitionWorker/db/db.csv')
        # lines = self.db['file_dir'].values.tolist()
        # TODO: index 값 저장! index 기반으로 유저 판단 -> 나중에 내뱉는 index 값에서 판단할 것!
        self.db_count = 0
        self.refImgs = []
        self.refVecs = []
        self.data = b''

    def find_Match(self, cnnVec):
        # Use L2 Distance
        findDist = lambda vec1, vec2: np.sum((vec1 - vec2) ** 2)
        l2_Dists = [findDist(cnnVec, vec) for vec in self.refVecs]
        # TODO: numpy argmin 은 경우에 따라 list 로 들어옴, 해당 경우에 대한 case check!
        index = np.argmin(l2_Dists)
        return index, l2_Dists[index]

    def inloop_recieveFile(self, socket, addr, req_name):
        # Send Ready Signal
        socket.sendall('Ready'.encode())

        # Receive filename
        msg = socket.recv(4096)
        time_stamp = msg.decode('utf-8')[:-4]
        log.debug(time_stamp)

        # Receive File Data
        while True:
            # get file bytes
            data = socket.recv(4096)
            self.data = self.data + data
            if not data:
                break

        temp_data = self.data
        # write bytes on file
        frame = cv2.imdecode(np.frombuffer(self.data, np.uint8), -1)

        # send API server image bytes
        data = {'req_addr': req_name,
                'image_packet': self.data}
        # requests.post('http://127.0.0.1:5000/image_packet', json=data)
        log.debug('send image packet')

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        log.debug(frame_gray.shape)

        faces = self.faceCascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100)
        )

        log.debug(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



        ##
        if len(faces):
            faceROI = frame_gray[y:y + h, x:x + w]
            faceROI = cv2.resize(faceROI, (128, 128))
            faceROI = cv2.equalizeHist(faceROI)
            log.debug(faceROI)

            cnnOut = cnnpRun(faceROI)

            if self.db_count == 0:
                headers = ['file_dir', 'name', 'age']

                image = Image.open(io.BytesIO(temp_data))
                img_name = str(self.db_count) + '.jpg'
                file_dir = '/home/haze/다운로드/apache-tomcat-7.0.92/webapps/HAZE/img/' + img_name
                image.save(file_dir, "JPEG")


                # TODO: byte 파일 저장 로직 여기 추가!
                image = Image.open(io.BytesIO(temp_data))
                img_dir = 'FaceRecognitionWorker/db/' + str(self.db_count) + '.jpg'
                image.save(img_dir, "JPEG")

                name = input('name: ')
                age = input('age: ')
                fields = [img_dir, name, age]

                with open('FaceRecognitionWorker/db/db.csv', 'w') as myfile:
                    writer = csv.writer(myfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(headers)
                    writer.writerow(fields)

                self.db_count = self.db_count+1
                self.refVecs.append(cnnOut)

            else:

                inx, minval = self.find_Match(cnnOut)
                log.debug("{}: {}".format(inx, minval))
                log.debug("{} : {} : {}".format(*self.db.ix[inx]))
                if minval > 1.0:
                    # status 변경 기준으로 가정
                    log.debug(type(cnnOut))
                    log.debug(type(addr))

                    # TODO: byte 파일 저장 로직 여기 추가!
                    image = Image.open(io.BytesIO(temp_data))
                    img_name = str(self.db_count) + '.jpg'
                    file_dir = '/home/haze/다운로드/apache-tomcat-7.0.92/webapps/HAZE/img/' + img_name
                    image.save(file_dir, "JPEG")

                    image = Image.open(io.BytesIO(temp_data))
                    img_dir = 'FaceRecognitionWorker/db/'+str(self.db_count)+'.jpg'
                    image.save(img_dir, "JPEG")

                    name = input('name: ')
                    age = input('age: ')
                    fields = [img_dir, name, age]
                    with open('FaceRecognitionWorker/db/db.csv', 'a') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerow(fields)

                    self.db_count = self.db_count+1
                    self.refVecs.append(cnnOut)
                    # 인식 성공했을 경우 1초 delay
                    # time.sleep(1)
        self.data = b''

    def main(self, HOST, PORT, requester_name):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(5)

        while True:

            try:
                requester_socket = s.accept()
                log.debug(requester_socket)
                self.inloop_recieveFile(*requester_socket, requester_name)

            except Exception as e:
                log.debug(str(e))
                log.debug('pass')
                continue
        # s.close()

        # self.inloop_Match(self, requester_socket, requester_name)

    def run(self):
        # TODO: 향후 아래의 cv2.VideoCapture 같이 이더넷 소켓 or 시리얼 통신 소켓을 열어서 A, B 로 부터 받아올것
        # TODO: 두 requester 에 대해서, 자원 공유를 할 필요가 없으니 multi processing 으로 병렬로 돌림
        # TODO: requester 들이 증분 될 경우.. 대책 필요
        self.main(HOST, PORT, 'REQUESTER1') # White
        # socket from requester A
        # socket from requester B
        # p2 = multiprocessing.Process(target=self.main, args=(HOST, PORT+1, 'REQUESTER2', )) # black
        # p2.start()


if __name__ == "__main__":
    _face_worker = FaceRecognitionWorker()
    _face_worker.run()

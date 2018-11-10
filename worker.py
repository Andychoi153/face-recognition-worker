import cv2
import numpy as np
import requests
import json
import pandas as pd
import socket
import multiprocessing
import time

from FaceRecognitionWorker.log import log

from cnnp import cnnpRun


HOST = ""
PORT = 9999


class FaceRecognitionWorker:

    def __init__(self):
        ##
        self.faceCascade = cv2.CascadeClassifier('FaceRecognitionWorker/config/haarcascade_frontalface_alt2.xml')

        ##
        # TODO: DB 형식의 csv 파일 포맷 생성
        self.db = pd.read_csv('FaceRecognitionWorker/db/db.csv')
        lines = self.db['file_dir'].values.tolist()
        # TODO: index 값 저장! index 기반으로 유저 판단 -> 나중에 내뱉는 index 값에서 판단할 것!

        self.refImgs = []
        self.refVecs = []
        self.data = b''
        for line in lines:
            temp = cv2.imread(line)

            self.refImgs.append(temp)

            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
            faces = self.faceCascade.detectMultiScale(
                temp,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            if len(faces) == 0:
                log.debug('{}: No Face!'.format(line))
            
                self.refVecs.append(np.zeros(512))
            else:
                log.debug('{}: Okay'.format(line))
                (x, y, w, h) = faces[0]
                faceROI = temp[y:y + h, x:x + w]
                faceROI = cv2.resize(faceROI, (128, 128))
                faceROI = cv2.equalizeHist(faceROI)
            
                log.debug(faceROI)
                cnnOut = cnnpRun(faceROI)
                log.debug(cnnOut)
                self.refVecs.append(cnnOut)

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

            # write bytes on file
        frame = cv2.imdecode(np.frombuffer(self.data, np.uint8), -1)
        log.debug(frame.shape)
        self.data = b''

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

            inx, minval = self.find_Match(cnnOut)
            log.debug("{}: {}".format(inx, minval))
            log.debug("{} : {} : {}".format(*self.db.ix[inx]))

            if minval < 0.9:
                # status 변경 기준으로 가정
                log.debug(type(cnnOut))
                log.debug(type(addr))
                data = {'req_addr': req_name,
                        'data': {'hash': hash(str(cnnOut)),
                                 'sol': {'name': self.db.ix[inx][1],
                                         'age': str(self.db.ix[inx][2])}
                                 },
                        'time_stamp': time_stamp
                        }
                requests.post('http://127.0.0.1:5000/create_transaction_by_contract', json=data)
                log.debug(json.dumps(data))
                # 인식 성공했을 경우 1초 delay
                # time.sleep(1)

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
                s.close()
                break

            # self.inloop_Match(self, requester_socket, requester_name)

    def run(self):
        # TODO: 향후 아래의 cv2.VideoCapture 같이 이더넷 소켓 or 시리얼 통신 소켓을 열어서 A, B 로 부터 받아올것
        # TODO: 두 requester 에 대해서, 자원 공유를 할 필요가 없으니 multi processing 으로 병렬로 돌림
        # TODO: requester 들이 증분 될 경우.. 대책 필요

        # socket from requester A
        p1 = multiprocessing.Process(target=self.main, args=(HOST, PORT, 'REQUESTER1',))

        # socket from requester B
        # p2 = multiprocessing.Process(target=self.main, args=(HOST, PORT+1, 'REQUESTER2', ))

        p1.start()
        # p2.start()


if __name__ == "__main__":
    _face_worker = FaceRecognitionWorker()
    _face_worker.run()

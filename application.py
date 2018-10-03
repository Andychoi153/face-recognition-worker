from worker import FaceRecognitionWorker
from FaceRecognitionWorker.log import log

if __name__ == "__main__":
    log.debug('Face recognition worker run')

    _face_worker = FaceRecognitionWorker
    _face_worker.run()

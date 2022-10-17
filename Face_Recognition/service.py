from face_detect_runnable import FaceDetectRunnable
from face_recognize_runnable import FaceRecognitionRunnable
import bentoml
from bentoml.io import *
import cv2
import numpy as np


face_detect_runner = bentoml.Runner(FaceDetectRunnable, name="face_detect", models=[bentoml.onnx.get("face_detection:latest")])
face_recognize_runner = bentoml.Runner(FaceRecognitionRunnable, name="face_recognize_runner", models=[bentoml.onnx.get("face_recognition:latest")])

svc = bentoml.Service("face_recognition", runners=[face_detect_runner, face_recognize_runner])

input_spec = Multipart(image=Image(), user_id=Text())
output_spec = NumpyNdarray()


@svc.api(input=input_spec, output=output_spec)
async def recognize(image, user_id):
    np_img = np.array(image)
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    bboxes, kpss = await face_detect_runner.detect.async_run(cv_img)
    embedding = await face_recognize_runner.recognize.async_run(cv_img, kpss[0])
    return embedding

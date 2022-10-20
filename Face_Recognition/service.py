from face_detect_runnable import FaceDetectRunnable
from face_recognize_runnable import FaceRecognitionRunnable
import bentoml
from bentoml.io import JSON
import cv2
import numpy as np
import boto3
import io
from pydantic import BaseModel
from PIL import Image
import base64

face_detect_runner = bentoml.Runner(FaceDetectRunnable, name="face_detect", models=[bentoml.onnx.get("face_detection:latest")])
face_recognize_runner = bentoml.Runner(FaceRecognitionRunnable, name="face_recognize_runner", models=[bentoml.onnx.get("face_recognition:latest")])

svc = bentoml.Service("face_recognition", runners=[face_detect_runner, face_recognize_runner])

class UserFace(BaseModel):
    user_id : int
    image : str

input_spec = JSON(pydantic_model=UserFace)
output_spec = JSON()


@svc.api(input=input_spec, output=output_spec, route="/api/recognize")
async def recognize(input_data: UserFace, ctx:bentoml.Context):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(input_data.image)))
    except:
        ctx.response.status_code = 400
        return {"message": "Unable to decode the image."}
    user_id = str(input_data.user_id)
    np_img = np.array(image)
    cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    bboxes, kpss = await face_detect_runner.detect.async_run(cv_img)

    # 얼굴이 검출 되지 않았을 때
    if len(kpss) == 0:
        ctx.response.status_code = 400
        return {"message": "No face"}
    # 여러 얼굴이 검출 되었을 때
    if len(kpss) > 1:
        ctx.response.status_code = 400
        return {"message": "Too many face"}

    embedding = await face_recognize_runner.recognize.async_run(cv_img, kpss[0])
    str_embedding = np.array2string(embedding)
    byte_embedding = io.BytesIO(str_embedding.encode())
    s3 = boto3.client('s3')
    s3.upload_fileobj(byte_embedding, 'everyonepick-ai-face-embedding-bucket', f"face{user_id}_embedding.txt")
    return {"message":"Ok", "data":{"user_id":user_id, "face_embedding":embedding}}

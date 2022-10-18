from bentoml.exceptions import *
import bentoml
import onnx
import boto3
import os

home_path = os.getenv("HOME")
download_path = os.path.join(home_path, 'model')

detection_model_file = 'scrfd_10g_bnkps.onnx'
recognition_model_file = 'glintr100.onnx'

detection_model_name = "face_detection"
recognition_model_name = "face_recognition"

if not os.path.exists(download_path):
    os.makedirs(download_path)

if not os.path.exists(os.path.join(download_path, detection_model_file)):
    client = boto3.client('s3', region_name='ap-northeast-2')
    client.download_file('everyonepick-ai-model-bucket', detection_model_file, os.path.join(download_path, detection_model_file))

if not os.path.exists(os.path.join(download_path, recognition_model_file)):
    client = boto3.client('s3', region_name='ap-northeast-2')
    client.download_file('everyonepick-ai-model-bucket', recognition_model_file, os.path.join(download_path, recognition_model_file))

try:
    bentoml.onnx.load_model(detection_model_name)
except NotFound:
    detection_model = onnx.load(os.path.join(download_path, detection_model_file))
    bentoml.onnx.save_model(detection_model_name, detection_model)

try:
    bentoml.onnx.load_model(recognition_model_name)
except NotFound:
    recognition_model = onnx.load(os.path.join(download_path, recognition_model_file))
    bentoml.onnx.save_model(recognition_model_name, recognition_model)
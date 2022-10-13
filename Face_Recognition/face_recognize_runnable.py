import bentoml
from bentoml.io import *
import numpy as np
import cv2
import onnx
import onnxruntime
import face_align


class FaceRecognitionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model_file = "../glintr100.onnx"
        self.session = onnxruntime.InferenceSession(self.model_file, None)
        self.session.set_providers(['CPUExecutionProvider'])
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    @bentoml.Runnable.method(batchable=False)
    def recognize(self, img, kps):
        aimg = face_align.norm_crop(img, kps)
        embedding = self.get_feat(aimg).flatten()
        return embedding


# face_recognize_runner = bentoml.Runner(FaceRecognitionRunnable, name="face_recognize")
# svc = bentoml.Service("face_recognizer", runners=[face_recognize_runner])
#
# input_spec = Multipart(img=Image(), kps=NumpyNdarray())
#
#
# @svc.api(input=input_spec, output=NumpyNdarray())
# async def recognize(img, kps):
#     np_img = np.array(img)
#     cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
#     embedding = await face_recognize_runner.recognize.async_run(cv_img, kps)
#     return embedding

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)

# then print all-to-all face similarity
feats = []
for face in faces:
    embedding = face.normed_embedding
    feats.append(embedding)
    # print(len(embedding))
feats = np.array(feats, dtype=np.float32)
sims = np.dot(feats, feats.T)
print(sims)
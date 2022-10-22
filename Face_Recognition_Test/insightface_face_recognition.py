import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

group = ins_get_image('group')
group_faces = app.get(group)
img1 = ins_get_image('target1')
face1 = app.get(img1)
img2 = ins_get_image('target2')
face2 = app.get(img2)
rimg = app.draw_on(group, group_faces)
cv2.imwrite("./group_output.jpg", rimg)
rimg = app.draw_on(img1, face1)
cv2.imwrite("./target1_output.jpg", rimg)
rimg = app.draw_on(img2, face2)
cv2.imwrite("./target2_output.jpg", rimg)


def compute_sim(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

group_feats = []
for face in group_faces:
    embedding = face.normed_embedding
    group_feats.append(embedding)

feat1 = face1[0].normed_embedding
feat2 = face2[0].normed_embedding
for feat in group_feats:
    print(compute_sim(feat1, feat))

for feat in group_feats:
    print(compute_sim(feat2, feat))
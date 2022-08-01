from collections import defaultdict, Counter
import cv2
from insightface.app import FaceAnalysis
import numpy as np
from numpy.linalg import norm as l2norm

''' 
사용자 별로 선호하는 사진 선택 결과 예시
id:A → group1, group2
id:B → group1
id:C → group3 
id:D → group1, group3
'''

# insightface Resnet100_Glint360k 얼굴 분석기 모델
face_app = FaceAnalysis('antelopev2')
face_app.prepare(ctx_id=0, det_size=(640, 640))

group_photo_path = ["../img_data/group1.jpeg", "../img_data/group2.jpeg", "../img_data/group3.jpeg"]
user_choices = defaultdict(list)

# user_choices -> {'A': [0, 1], 'B': [0], 'C': [2], 'D': [0, 2]})
user_choices['A'].extend([0, 1])
user_choices['B'].append(0)
user_choices['C'].append(2)
user_choices['D'].extend([0, 2])

# 가장 많은 선택을 받은 사진을 찾는 함수
def find_base_photo(user_choices):
    # 아무도 선택 하지 않은 경우
    if len(user_choices) == 0:
        return -1

    result = []
    base_photo = []

    for choices in user_choices.values():
        result += choices

    # 사진 별 선택 수
    photo_choices = Counter(result).most_common()
    # base 사진, 받은 선택 수
    max_choice = photo_choices[0][1]

    # base 사진의 선택 수가 중복 되는지 확인
    for photo, choice in photo_choices:
        if choice == max_choice:
            base_photo.append(photo)

    ''' base 사진이 여러개 가능할 경우 후보군 중에 어떤 base 사진이 가장 최적인지 계산하는 코드 추가 에정
    ex. 얼굴 각도 계산을 통해 가장 자연스럽게 얼굴이 합성될 수 있는 케이스 찾기 '''

    return base_photo


# face swap이 필요한 user_id와 source 사진을 찾는 함수
def list_of_face_swap(user_choices, base_photo):
    ''' 현재 사용자 별 선택 사진 예시로는 group3 사진에서의 사용자 C얼굴을 group1 사진에 합성하면 되지만,
    만약 사용자 C가 group2, 3을 선택했다면 group2, 3 사진 중에서 어떤 얼굴을 합성할 지 결정하는 과정 추가 예정
    ex. 얼굴 각도 계산을 통해 가장 자연스럽게 얼굴이 합성될 수 있는 케이스 찾기 '''

    source_target_list = []

    for user, choices in user_choices.items():
        if base_photo not in choices:
            # 임시로 첫 번째 선택 사진을 target 사진으로 결정
            source_target_list.append((user, choices[0]))

    return source_target_list


def face_analysis(img_path):
    img = cv2.imread(img_path)
    img = img[:, :, ::-1]
    faces = face_app.get(img)
    return faces


def get_embeddings(faces):
    embeddings = []
    for face in faces:
        embeddings.append(face['embedding'])

    return embeddings


def face_recognition(group_embeddings, target_embedding):
    # 임베딩 정규화
    normed_target_embedding = target_embedding / l2norm(target_embedding)
    normed_group_embeddings = []
    for embedding in group_embeddings:
        normed_group_embeddings.append(embedding / l2norm(embedding))

    normed_group_embeddings = np.array(normed_group_embeddings, dtype=np.float32)
    # 코사인 유사도 계산
    sims = np.dot(normed_target_embedding, normed_group_embeddings.T)

    return sims

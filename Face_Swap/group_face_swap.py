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


# insightface 모델로 얼굴 정보 분석하는 함수
# bbox, kps, det_score, landmark_3d_69, pose, landmark_2d_106, gender, embedding 포함
def face_analysis(img_path):
    img = cv2.imread(img_path)
    img = img[:, :, ::-1]
    faces = face_app.get(img)
    return faces


# 얼굴 임베딩 리스트 반환하는 함수
def get_embeddings(faces):
    embeddings = []
    for face in faces:
        embeddings.append(face['embedding'])

    return embeddings


# target_embedding과 그룹 사진 얼굴들과의 유사도 측정하는 함수
def compute_face_similarity(group_embeddings, target_embedding):
    # 임베딩 정규화
    normed_target_embedding = target_embedding / l2norm(target_embedding)
    normed_group_embeddings = []
    for embedding in group_embeddings:
        normed_group_embeddings.append(embedding / l2norm(embedding))

    normed_group_embeddings = np.array(normed_group_embeddings, dtype=np.float32)
    # 코사인 유사도 계산
    sims = np.dot(normed_target_embedding, normed_group_embeddings.T)

    return sims


# landmark 데이터 타입을 int로 형 변환 하는 함수
def cast_int_landmarks(landmarks):
    result = []

    for i in range(landmarks.shape[0]):
        point = tuple(landmarks[i])
        x = int(point[0])
        y = int(point[1])
        result.append((x, y))

    return result


# nparray에서 0번째 값 반환
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


# img에서 특정 얼굴의 들로네 삼각망 구하기
def get_delaunay_traingles(img, mask, landmarks):
    points = np.array(landmarks, np.int32)
    # 얼굴 특징점 경계선 추출
    convexhull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convexhull, 255)
    # 얼굴 특징점 경계선을 따라 얼굴 이미지 추출
    face_image = cv2.bitwise_and(img, img, mask=mask)
    # 얼굴 들로네 삼각망 구축
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # 들로네 삼각망의 triangles 인덱스 구하기
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles


if __name__ == "__main__":
    # 지금 설정한 케이스 예시에는 베이스 사진이 하나기 때문에 index 0으로 설정
    target_photo = find_base_photo(user_choices)[0]
    # 베이스 사진의 얼굴 정보 분석
    target_faces = face_analysis(group_photo_path[target_photo])
    # 베이스 사진 얼굴들의 임베딩 가져오기
    target_embeddings = get_embeddings(target_faces)

    # face swap이 들어가야 되는 사용자와 해당 사용자가 선택한 사진 번호 반환
    face_swap_list = list_of_face_swap(user_choices, target_photo)

    # db에 저장된 사용자의 얼굴 정보 가져오기 (아래 코드는 임시 작성)
    user_face = face_analysis('../img_data/jh.jpeg')

    # user : C, choice : 2
    for user, choice in face_swap_list:
        # 사용자의 기본 임베딩 가져오기
        user_embedding = user_face[0]['embedding']
        # 베이스 사진에서 사용자 얼굴의 index 찾기
        target_user_index = compute_face_similarity(target_embeddings, user_embedding).argmax()
        # 베이스 사진에서 사용자 얼굴의 landmarks 찾기
        target_user_landmarks = target_faces[target_user_index]['landmark_2d_106']

        # 사용자가 선택한 사진의 얼굴 정보 분석
        source_faces = face_analysis(group_photo_path[choice])
        # 사용자가 선택한 사진 얼굴들의 임베딩 가져오기
        source_embeddings = get_embeddings(source_faces)
        # 사용자가 선택한 사진에서 사용자 얼굴의 index 찾기
        source_user_index = compute_face_similarity(source_embeddings, user_embedding).argmax()
        # 사용자가 선택한 사진에서 사용자 얼굴의 landmarks 찾기
        source_user_landmarks = source_faces[source_user_index]['landmark_2d_106']

        face_swap('../img_data/jh.jpeg', group_photo_path[0], source_user_landmarks, target_user_landmarks)
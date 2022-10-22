from collections import defaultdict, Counter
import cv2
from insightface.app import FaceAnalysis
import numpy as np
from numpy.linalg import norm as l2norm
import matplotlib.pyplot as plt

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

group_img_path = []
for i in range(1, 8):
    group_img_path.append(f"../img_data/{i}.JPG")

user_choices = defaultdict(list)
user_profiles = {}

# user_choices -> {'A': [0, 1], 'B': [0], 'C': [2], 'D': [0, 2]})
user_choices['ys'].extend([2, 6])
user_choices['js'].extend([4, 5, 6])
user_choices['es'].extend([1, 2, 6])
user_choices['jy'].extend([4, 5])
user_choices['sj'].extend([3, 4, 5, 6])
user_choices['je'].extend([2, 4, 5])
user_choices['sy'].extend([2, 3, 5])
user_choices['jh'].extend([3])


user_profiles['ys'] = "../img_data/ys.jpeg"
user_profiles['js'] = "../img_data/js.jpeg"
user_profiles['es'] = "../img_data/es.jpeg"
user_profiles['jy'] = "../img_data/jy.jpeg"
user_profiles['sj'] = "../img_data/sj.jpeg"
user_profiles['je'] = "../img_data/je.jpeg"
user_profiles['sy'] = "../img_data/sy.jpeg"
user_profiles['jh'] = "../img_data/jh.jpeg"


# 가장 많은 선택을 받은 사진의 인덱스를 찾는 함수
def find_base_img_index(user_choices):
    # 아무도 선택 하지 않은 경우
    if len(user_choices) == 0:
        return -1

    result = []
    base_img = []

    for choices in user_choices.values():
        result += choices

    # 사진 별 선택 수
    img_choices = Counter(result).most_common()
    # base 사진, 받은 선택 수
    max_choice = img_choices[0][1]

    # base 사진의 선택 수가 중복 되는지 확인
    for img, choice in img_choices:
        if choice == max_choice:
            base_img.append(img)

    ''' base 사진이 여러개 가능할 경우 후보군 중에 어떤 base 사진이 가장 최적인지 계산하는 코드 추가 에정
    ex. 얼굴 각도 계산을 통해 가장 자연스럽게 얼굴이 합성될 수 있는 케이스 찾기 '''

    return base_img


# face swap이 필요한 user_id와 source 사진을 찾는 함수
def list_of_face_swap(user_choices, base_img):
    ''' 현재 사용자 별 선택 사진 예시로는 group3 사진에서의 사용자 C얼굴을 group1 사진에 합성하면 되지만,
    만약 사용자 C가 group2, 3을 선택했다면 group2, 3 사진 중에서 어떤 얼굴을 합성할 지 결정하는 과정 추가 예정
    ex. 얼굴 각도 계산을 통해 가장 자연스럽게 얼굴이 합성될 수 있는 케이스 찾기 '''

    source_target_list = []

    for user, choices in user_choices.items():
        if base_img not in choices:
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

def triangulation_two_faces(lines_space_mask, indexes_triangles, source_img, source_landmarks, target_landmarks, target_new_face):
    # source 얼굴 들로네 삼각망을 target 얼굴 들로네 삼각망 일치하도록 변형
    for triangle_index in indexes_triangles:
        tr1_pt1 = source_landmarks[triangle_index[0]]
        tr1_pt2 = source_landmarks[triangle_index[1]]
        tr1_pt3 = source_landmarks[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = source_img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        # lines_space = cv2.bitwise_and(source_img, source_img, mask=lines_space_mask)

        tr2_pt1 = target_landmarks[triangle_index[0]]
        tr2_pt2 = target_landmarks[triangle_index[1]]
        tr2_pt3 = target_landmarks[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # 변형된 triangles 연결해서 얼굴 구축
        img2_new_face_rect_area = target_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        target_new_face[y: y + h, x: x + w] = img2_new_face_rect_area


def face_swap(source_img, target_img, f_source_landmarks, f_target_landmarks):
    # source_img = cv2.imread(source_img_path)
    # source_img = source_img[:, :, ::-1]

    # target_img = cv2.imread(target_img_path)
    # target_img = target_img[:, :, ::-1]

    source_img_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(source_img_gray)
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    height, width, channels = target_img.shape
    target_new_face = np.zeros((height, width, channels), np.uint8)

    source_landmarks = cast_int_landmarks(f_source_landmarks)
    target_landmarks = cast_int_landmarks(f_target_landmarks)

    # target 얼굴 특징점 경계선 검출
    points = np.array(target_landmarks, np.int32)
    target_convexhull = cv2.convexHull(points)

    indexes_triangles = get_delaunay_traingles(source_img, mask, source_landmarks)

    lines_space_mask = np.zeros_like(source_img_gray)

    triangulation_two_faces(lines_space_mask, indexes_triangles, source_img, source_landmarks, target_landmarks, target_new_face)

    # destination 얼굴에 source 얼굴 합성하기
    img2_face_mask = np.zeros_like(target_img_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, target_convexhull, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)

    img2_head_noface = cv2.bitwise_and(target_img, target_img, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, target_new_face)

    # 합성이 자연스럽도록 색 조정
    (x, y, w, h) = cv2.boundingRect(target_convexhull)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamlessclone = cv2.seamlessClone(result, target_img, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    return seamlessclone


''' face swap 함수 테스트 (사용자 선택 고려 O)'''
if __name__ == "__main__":
    # 지금은 베이스 사진이 여러개가 가능할 경우를 무시함
    target_img_index = find_base_img_index(user_choices)[0]
    target_img_path = group_img_path[target_img_index]
    target_img = cv2.imread(target_img_path)
    target_img = target_img[:, :, ::-1]
    target_faces = face_analysis(target_img_path)
    target_embeddings = get_embeddings(target_faces)

    face_swap_list = list_of_face_swap(user_choices, target_img_index)

    for user, choice in face_swap_list:
        user_img_path = user_profiles[user]
        source_img_path = group_img_path[choice]
        source_img = cv2.imread(source_img_path)
        source_img = source_img[:, :, ::-1]

        user_face = face_analysis(user_img_path)
        source_faces = face_analysis(source_img_path)

        user_embedding = user_face[0]['embedding']
        source_embeddings = get_embeddings(source_faces)

        target_user_face_index = compute_face_similarity(target_embeddings, user_embedding).argmax()
        source_user_face_index = compute_face_similarity(source_embeddings, user_embedding).argmax()

        target_face_landmarks = target_faces[target_user_face_index]['landmark_2d_106']
        source_face_landmarks = source_faces[source_user_face_index]['landmark_2d_106']

        swap_result = face_swap(source_img, target_img, source_face_landmarks, target_face_landmarks)
        target_img = swap_result
        plt.imshow(target_img)
        plt.show()


# ''' face swap 함수 테스트 (사용자 선택 고려 X)'''
# if __name__ == "__main__":
#     source_img_path = '../img_data/3.JPG'
#     target_img_path = '../img_data/6.JPG'
#     user_img_path = '../img_data/ys.jpeg'
#
#     source_img = cv2.imread(source_img_path)
#     source_img = source_img[:, :, ::-1]
#     target_img = cv2.imread(target_img_path)
#     target_img = target_img[:, :, ::-1]
#
#     user_face = face_analysis(user_img_path)
#     source_faces = face_analysis(source_img_path)
#     target_faces = face_analysis(target_img_path)
#
#     user_embedding = user_face[0]['embedding']
#     source_embeddings = get_embeddings(source_faces)
#     target_embeddings = get_embeddings(target_faces)
#
#     source_user_index = compute_face_similarity(source_embeddings, user_embedding).argmax()
#     source_user_landmarks = source_faces[source_user_index]['landmark_2d_106']
#
#     target_user_index = compute_face_similarity(target_embeddings, user_embedding).argmax()
#     target_user_landmarks = target_faces[target_user_index]['landmark_2d_106']
#
#     result = face_swap(source_img, target_img, source_user_landmarks, target_user_landmarks)
#     plt.imshow(result)
#     plt.show()

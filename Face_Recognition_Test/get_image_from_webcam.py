import os.path

import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles


cap = cv2.VideoCapture(0)

E_count = 0
W_count = 0
S_count = 0
N_count = 0
NW_count = 0
SE_count = 0
NE_count = 0
SW_count = 0

while cap.isOpened():
    success, image = cap.read()

    start = time.time()
    if not success:
        print("Ignoring empty camera frame.")
        break

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    # Flip the image horizontally for a later selfie-view display
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Get the result
    mesh_results = face_mesh.process(image)
    detect_results = face_detection.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if detect_results.detections:
        for detection in detect_results.detections:
            mp_drawing.draw_detection(image, detection)

    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                # idx: 1 -> Tip of the nose
                # idx: 61 -> Left corner of the mouth
                # idx: 291 -> Right corner of the mouth
                # idx: 199 -> Chin
                # idx: 33 -> Left corner of the left eye
                # idx: 263 -> Right corner of the right eye
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 1000)

                    # 정규화 되어 있기 때문에 이미지의 높이, 너비 반영하기
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z * 5])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            # print(dist_matrix)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)


            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360



            # See where the user's head tilting
            # East
            if y > 20 and -5 < x < 5:
                if E_count < 10:
                    cv2.imwrite(os.path.join("../output", "E_{}.jpg".format(E_count)), image)
                    E_count += 1
                text = "Looking East"
            # West
            elif y < -20 and -5 < x < 5:
                if W_count < 10:
                    cv2.imwrite(os.path.join("../output", "W_{}.jpg".format(W_count)), image)
                    W_count += 1
                text = "Looking West"
            # South
            elif x < -20 and -5 < y < 5:
                if S_count < 10:
                    cv2.imwrite(os.path.join("../output", "S_{}.jpg".format(S_count)), image)
                    S_count += 1
                text = "Looking South"
            # North
            elif x > 20 and -5 < y < 5:
                if N_count < 10:
                    cv2.imwrite(os.path.join("../output", "N_{}.jpg".format(N_count)), image)
                    N_count += 1
                text = "Looking North"
            # North-East
            elif x > 20 and y > 20:
                if NE_count < 10:
                    cv2.imwrite(os.path.join("../output", "NE_{}.jpg".format(NE_count)), image)
                    NE_count += 1
                text = "Looking North-East"
            # South-East
            elif x < -20 and y > 20:
                if SE_count < 10:
                    cv2.imwrite(os.path.join("../output", "SE_{}.jpg".format(SE_count)), image)
                    SE_count += 1
                text = "Looking South-East"
            # South-West
            elif x < -20 and y < -20:
                if SW_count < 10:
                    cv2.imwrite(os.path.join("../output", "SW_{}.jpg".format(SW_count)), image)
                    SW_count += 1
                text = "Looking South-West"
            # North-West
            elif x > 20 and y < -20:
                if NW_count < 10:
                    cv2.imwrite(os.path.join("../output", "NW_{}.jpg".format(NW_count)), image)
                    NW_count += 1
                text = "Looking North-East"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
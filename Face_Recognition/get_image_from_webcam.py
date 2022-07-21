import cv2

# 기본적으로 웹캠은 device id: 0
cap = cv2.VideoCapture(0)

while(True):
    # ret: 카메라 정상 연결 여부 True / False
    # frame: 현재 프레임의 정보
    ret, frame = cap.read()    # Read 결과와 frame

    if(ret) :
        cv2.imshow('head_pose_estimation', frame)    # 컬러 화면 출력
        if cv2.waitKey(1) == ord('q'):
            break

# 카메라 메모리 해제
cap.release()
# 모든 창 닫기
cv2.destroyAllWindows()
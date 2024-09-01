import cv2
import mediapipe as mp

# MediaPipe와 OpenCV 초기화
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 랜드마크 그리기 설정
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))  # 점의 크기와 색상 설정

# 비디오 캡처 설정 (웹캠 또는 비디오 파일)
# cap = cv2.VideoCapture(0)  # 웹캠 사용
cap = cv2.VideoCapture("./video/smile.mp4")

# 입술의 양끝 (좌우) 랜드마크 인덱스
LEFT_LIP = 61   # 입술 왼쪽 끝
RIGHT_LIP = 291  # 입술 오른쪽 끝

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 이미지 색상 변환 및 얼굴 랜드마크 감지
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 이미지 색상 복원
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 입술 좌우 끝의 X 좌표 추출
                left_lip_x = face_landmarks.landmark[LEFT_LIP].x
                right_lip_x = face_landmarks.landmark[RIGHT_LIP].x

                # 이미지 크기에 맞춰 좌표 변환
                left_lip_pixel_x = int(left_lip_x * image.shape[1])
                right_lip_pixel_x = int(right_lip_x * image.shape[1])

                # 입술 양쪽 끝의 가로 거리 계산
                lip_width = abs(right_lip_pixel_x - left_lip_pixel_x)
                print(lip_width)
                # 기준값을 설정하여 입술이 양옆으로 충분히 벌어졌는지 확인 ('I' 발음)
                if lip_width > 390:  # 이 값을 조정하여 'I' 발음 감지 민감도를 조절
                    cv2.putText(image, "OK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 입술 랜드마크 연결선 그리기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LIPS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
                )

        # 이미지 표시
        cv2.imshow('MediaPipe Face Mesh - Lip Detection', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(5) == ord('q'):
            break

# 비디오 캡처 및 윈도우 해제
cap.release()
cv2.destroyAllWindows()

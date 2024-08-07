import cv2
import mediapipe as mp

# MediaPipe와 OpenCV 초기화
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 랜드마크 그리기 설정
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))  # 점의 크기와 색상 설정

# 비디오 캡처 설정 (웹캠 또는 비디오 파일)
cap = cv2.VideoCapture(0)  # 웹캠 사용
# cap = cv2.VideoCapture("./video/face2.mp4")  # 비디오 파일 사용

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
                # 랜드마크 그리기
                for i, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(image, (x, y), drawing_spec.circle_radius, (0, 255, 0), drawing_spec.thickness)
                    cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

                # 랜드마크 연결선 그리기
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
                )

        # 이미지 표시
        cv2.imshow('MediaPipe Face Mesh', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(5) == ord('q'):
            break

# 비디오 캡처 및 윈도우 해제
cap.release()
cv2.destroyAllWindows()

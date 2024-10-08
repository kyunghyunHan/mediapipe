import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))  # 점의 크기와 색상을 설정
cap = cv2.VideoCapture(0)
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

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # 입술 왼쪽 끝 (61번)과 오른쪽 끝 (291번) 좌표 추출
        left_lip_idx = 61
        right_lip_idx = 291

        left_lip_landmark = face_landmarks.landmark[left_lip_idx]
        right_lip_landmark = face_landmarks.landmark[right_lip_idx]

        left_lip_x = int(left_lip_landmark.x * image.shape[1])
        left_lip_y = int(left_lip_landmark.y * image.shape[0])

        right_lip_x = int(right_lip_landmark.x * image.shape[1])
        right_lip_y = int(right_lip_landmark.y * image.shape[0])

        print(f"왼쪽 입술 끝 좌표: ({left_lip_x}, {left_lip_y})")
        print(f"오른쪽 입술 끝 좌표: ({right_lip_x}, {right_lip_y})")
        print(f"입술 세로길이:({right_lip_x-left_lip_x})")

        #입술 길이 82
        #최대길이 127 

        # 입술 끝 좌표 표시
        cv2.circle(image, (left_lip_x, left_lip_y), drawing_spec.circle_radius, (255, 0, 0), drawing_spec.thickness)
        cv2.circle(image, (right_lip_x, right_lip_y), drawing_spec.circle_radius, (255, 0, 0), drawing_spec.thickness)

        # 입술 부분의 랜드마크만 점으로 그리기
        lip_indices = set([idx for connection in mp_face_mesh.FACEMESH_LIPS for idx in connection])
        for idx in lip_indices:
          landmark = face_landmarks.landmark[idx]
          x = int(landmark.x * image.shape[1])
          y = int(landmark.y * image.shape[0])
          cv2.circle(image, (x, y), drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh - Lips Only (Dots)', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()



import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 전문가와 자신의 영상 경로
expert_video_path = "./video/face1.mp4"
my_video_path = 0  # 카메라 입력 사용

# 전문가 영상과 자신의 영상 각각 열기
cap_expert = cv2.VideoCapture(expert_video_path)
cap_my = cv2.VideoCapture(my_video_path)

# 입술 전체의 랜드마크 인덱스 (Mediapipe 기준)
lip_landmark_indices = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # 바깥쪽 입술
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324  # 안쪽 입술
]

# 유클리드 거리 계산 함수
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# 비율 유지 리사이즈 함수
def resize_with_aspect_ratio(image, target_height):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image

with mp_face_mesh.FaceMesh(
    max_num_faces=2,  # 최대 2개의 얼굴 인식
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap_expert.isOpened() and cap_my.isOpened():
        success_expert, image_expert = cap_expert.read()
        success_my, image_my = cap_my.read()

        if not success_expert or not success_my:
            print("Ignoring empty camera frame.")
            break

        # 두 영상의 최대 높이 설정 (조정할 높이)
        target_height = 480  # 원하는 높이 설정

        # 비율 유지하면서 리사이즈
        image_expert = resize_with_aspect_ratio(image_expert, target_height)
        image_my = resize_with_aspect_ratio(image_my, target_height)

        # 두 영상을 나란히 붙이기
        combined_image = np.hstack((image_expert, image_my))

        # Mediapipe를 위한 이미지 전처리
        combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(combined_image_rgb)

        # 각 영상의 입술 좌표 추출
        expert_lip_points = []
        my_lip_points = []

        # 두 얼굴 랜드마크 그리기
        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                for lip_idx in lip_landmark_indices:
                    x = face_landmarks.landmark[lip_idx].x
                    y = face_landmarks.landmark[lip_idx].y

                    # 랜드마크 점 그리기
                    cv2.circle(combined_image, (int(x * combined_image.shape[1]), int(y * combined_image.shape[0])), 2, (0, 255, 0), -1)

                    # 입술 좌표 추가 (왼쪽과 오른쪽 얼굴 인식 분리)
                    if idx == 0:  # 전문가 영상에서
                        expert_lip_points.append([x, y])
                    elif idx == 1:  # 내 영상에서
                        my_lip_points.append([x, y])

        # 코사인 유사도 계산 (입술 좌표가 있을 경우)
        if expert_lip_points and my_lip_points:
            # 평균 중앙 좌표 계산
            expert_lip_points = np.array(expert_lip_points)
            my_lip_points = np.array(my_lip_points)

            expert_center = np.mean(expert_lip_points, axis=0)
            my_center = np.mean(my_lip_points, axis=0)

            # 중앙 기준으로 상대적 위치 계산
            expert_relative = expert_lip_points - expert_center
            my_relative = my_lip_points - my_center

            # 유클리드 거리 계산
            dist = euclidean_distance(expert_relative.flatten(), my_relative.flatten())
            print(f"유클리드 거리: {dist:.4f}")

            # 유사도를 내 영상에 텍스트로 표시
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            color = (255, 0, 0)
            text = f"Euclidean Distance: {dist:.4f}"
            cv2.putText(combined_image, text, (50, 50), font, font_scale, color, font_thickness, cv2.LINE_AA)

        # 결과 화면 출력
        cv2.imshow('Expert vs My Video', combined_image)

        if cv2.waitKey(10) == ord('q'):
            break

# Release resources
cap_expert.release()
cap_my.release()
cv2.destroyAllWindows()

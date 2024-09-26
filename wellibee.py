import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 전문가와 자신의 영상 경로
expert_video_path = "./video/face1.mp4"
expert_video_path = "./video/smile.mp4"

my_video_path = "./video/face2.mp4"
my_video_path = 0

# 전문가 영상과 자신의 영상 각각 열기
cap_expert = cv2.VideoCapture(expert_video_path)
cap_my = cv2.VideoCapture(my_video_path)

# 입술 전체의 랜드마크 인덱스 (Mediapipe 기준)
lip_landmark_indices = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # 바깥쪽 입술
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324  # 안쪽 입술
]

# 코사인 유사도 계산 함수
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# 비율 유지 리사이즈 함수
def resize_with_aspect_ratio(image, target_width):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_height = int(target_width / aspect_ratio)
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap_expert.isOpened() and cap_my.isOpened():
        success_expert, image_expert = cap_expert.read()
        success_my, image_my = cap_my.read()

        if not success_expert or not success_my:
            print("Ignoring empty camera frame.")
            break

        # 두 영상의 크기 구하기
        expert_height, expert_width = image_expert.shape[:2]
        my_height, my_width = image_my.shape[:2]

        # 두 영상의 최대 너비 설정 (조정할 너비)
        target_width = 640  # 원하는 너비 설정

        # 비율 유지하면서 리사이즈
        image_expert = resize_with_aspect_ratio(image_expert, target_width)
        image_my = resize_with_aspect_ratio(image_my, target_width)

        # 크기 조정 후 최종 높이와 너비 구하기
        target_height = max(image_expert.shape[0], image_my.shape[0])
        image_expert = cv2.resize(image_expert, (target_width, target_height))
        image_my = cv2.resize(image_my, (target_width, target_height))

        # Mediapipe를 위한 이미지 전처리 (전문가와 내 영상 각각)
        image_expert_rgb = cv2.cvtColor(image_expert, cv2.COLOR_BGR2RGB)
        results_expert = face_mesh.process(image_expert_rgb)

        image_my_rgb = cv2.cvtColor(image_my, cv2.COLOR_BGR2RGB)
        results_my = face_mesh.process(image_my_rgb)

        # 각 영상의 입술 좌표 추출
        expert_lip_points = []
        my_lip_points = []

        # 전문가 영상에서 입술 좌표 추출
        if results_expert.multi_face_landmarks:
            for face_landmarks in results_expert.multi_face_landmarks:
                for idx in lip_landmark_indices:
                    x = int(face_landmarks.landmark[idx].x * target_width)
                    y = int(face_landmarks.landmark[idx].y * target_height)
                    expert_lip_points.append([x, y])
                    # 입술 점 그리기
                    cv2.circle(image_expert, (x, y), 2, (0, 255, 0), -1)
        else:
            print("전문가 영상에서 얼굴이 감지되지 않았습니다.")

        # 내 영상에서 입술 좌표 추출
        if results_my.multi_face_landmarks:
            for face_landmarks in results_my.multi_face_landmarks:
                for idx in lip_landmark_indices:
                    x = int(face_landmarks.landmark[idx].x * target_width)
                    y = int(face_landmarks.landmark[idx].y * target_height)
                    my_lip_points.append([x, y])
                    # 입술 점 그리기
                    cv2.circle(image_my, (x, y), 2, (0, 255, 0), -1)
        else:
            print("내 영상에서 얼굴이 감지되지 않았습니다.")
        # 코사인 유사도 계산 (입술 좌표가 있을 경우)
        if expert_lip_points and my_lip_points:
            expert_lip_points_flat = np.array(expert_lip_points).flatten()
            my_lip_points_flat = np.array(my_lip_points).flatten()
            cos_sim = cosine_similarity(expert_lip_points_flat, my_lip_points_flat)
            print(f"코사인 유사도: {cos_sim:.4f}")

            # 유사도를 내 영상에 텍스트로 표시
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            color = (255, 0, 0)
            text = f"Cosine Similarity: {cos_sim:.4f}"
            cv2.putText(image_my, text, (50, 50), font, font_scale, color, font_thickness, cv2.LINE_AA)

        # 두 영상을 나란히 붙이기
        combined_image = np.hstack((image_expert, image_my))

        # 결과 화면 출력
        cv2.imshow('Expert vs My Video', combined_image)

        if cv2.waitKey(10) == ord('q'):
            break

# Release resources
cap_expert.release()
cap_my.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp

from PIL import Image, ImageSequence
import time

def load_apng_frames(apng_path):
    apng = Image.open(apng_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(apng)]
    return frames

def resize_overlay(overlay, scale):
    overlay_resized = overlay.resize((int(overlay.width * scale), int(overlay.height * scale)), Image.LANCZOS)
    return overlay_resized

def remove_background(image, background_color=(255, 255, 255)):
    # Make background_color transparent
    image = image.convert("RGBA")
    data = np.array(image)
    
    # Convert the background color to an alpha mask
    red, green, blue = background_color
    mask = (data[..., :3] == [red, green, blue]).all(axis=2)
    data[mask] = [0, 0, 0, 0]  # Set background color to transparent
    
    return Image.fromarray(data)

def overlay_image(background, overlay, position):
    x, y = position
    bg_height, bg_width = background.shape[:2]
    ol_height, ol_width = overlay.shape[:2]

    if x >= bg_width or y >= bg_height:
        return background

    ol_crop_width = min(bg_width - x, ol_width)
    ol_crop_height = min(bg_height - y, ol_height)

    overlay_cropped = overlay[:ol_crop_height, :ol_crop_width]

    mask = overlay_cropped[..., 3:] / 255.0
    mask = np.repeat(mask, 3, axis=2)  # Expand mask to 3 channels
    overlay_rgb = overlay_cropped[..., :3]

    background[y:y+ol_crop_height, x:x+ol_crop_width] = \
        (1.0 - mask) * background[y:y+ol_crop_height, x:x+ol_crop_width] + \
        mask * overlay_rgb

    return background

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# APNG 파일 로드 및 크기 조정
apng_frames = load_apng_frames("./img/elephant.png")
apng_frames = [remove_background(frame) for frame in apng_frames]  # 배경 제거
scale_factor = 0.3  # APNG 크기 축소 비율

cap = cv2.VideoCapture("./video/face1.mp4")
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))

# APNG를 1초 동안만 실행하도록 설정
start_time = time.time()
duration = 10  # 1초

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    frame_index = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("비디오 파일의 끝에 도달했습니다.")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_lip_idx = 61
                right_lip_idx = 291

                left_lip_landmark = face_landmarks.landmark[left_lip_idx]
                right_lip_landmark = face_landmarks.landmark[right_lip_idx]

                left_lip_x = int(left_lip_landmark.x * image.shape[1])
                left_lip_y = int(left_lip_landmark.y * image.shape[0])

                right_lip_x = int(right_lip_landmark.x * image.shape[1])
                right_lip_y = int(right_lip_landmark.y * image.shape[0])

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
                

        # Check if the current time is within the 1-second window
        elapsed_time = time.time() - start_time
        if elapsed_time <= duration:
            # APNG 프레임을 비디오 프레임 위에 오버레이
            apng_frame = apng_frames[frame_index % len(apng_frames)]
            apng_frame_resized = resize_overlay(apng_frame, scale_factor)
            apng_frame_np = cv2.cvtColor(np.array(apng_frame_resized), cv2.COLOR_RGBA2BGRA)
            
            # APNG를 화면 중앙에 위치시키기 위해 위치 계산
            apng_x = (image.shape[1] - apng_frame_np.shape[1]) // 2
            apng_y = (image.shape[0] - apng_frame_np.shape[0]) // 2

            image = overlay_image(image, apng_frame_np, (apng_x, apng_y))

            frame_index += 1
        else:
            # 1초가 지나면 APNG를 제거하거나 대체 이미지로 변경
            pass

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh with APNG Overlay', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
cap.release()
cv2.destroyAllWindows()

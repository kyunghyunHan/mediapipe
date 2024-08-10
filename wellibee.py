import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))  # 점의 크기와 색상을 설정
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./video/face2.mp4")
mb_executed = False
default_lib_x = 0
default_lib_inner_y = 0
default_lib_outer_y = 0
face_y =0

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
     

    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # mp_drawing.draw_landmarks(
        #       image=image,
        #       landmark_list=face_landmarks,
        #       connections=mp_face_mesh.FACEMESH_CONTOURS,  # 얼굴 랜드마크의 연결선
        #       landmark_drawing_spec=drawing_spec,  # 랜드마크 점 스타일
        #       connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))  # 연결선 스타일
        #   )
        face_top_idx= 10
        face_buttom_idx= 152

        middle_outer_top_lib_idx =0
        middle_outer_buttom_lib_idx =17
        middle_inner_top_lip_idx = 13
        middle_inner_buttom_lip_idx = 14
        left_lip_idx = 61
        right_lip_idx = 291

        ## 얼굴 
        face_top_landmarks = face_landmarks.landmark[face_top_idx]
        face_buttom_landmarks = face_landmarks.landmark[face_buttom_idx]
        ## 입
        left_lip_landmark = face_landmarks.landmark[left_lip_idx]
        right_lip_landmark = face_landmarks.landmark[right_lip_idx]
        middle_outer_top_landmark = face_landmarks.landmark[middle_outer_top_lib_idx]
        middle_outer_buttom_landmark = face_landmarks.landmark[middle_outer_buttom_lib_idx]


        middle_inner_top_landmark = face_landmarks.landmark[middle_inner_top_lip_idx]
        middle_inner_buttom_landmark = face_landmarks.landmark[middle_inner_buttom_lip_idx]

        #얼굴
        face_top_x = int(face_top_landmarks.x * image.shape[1])
        face_top_y = int(face_top_landmarks.y * image.shape[0])
        face_buttom_x = int(face_buttom_landmarks.x * image.shape[1])
        face_buttom_y = int(face_buttom_landmarks.y * image.shape[0])
        #왼쪽 끝
        left_lip_x = int(left_lip_landmark.x * image.shape[1])
        left_lip_y = int(left_lip_landmark.y * image.shape[0])
        #가운데 바깥쪽 위
        middle_outer_top_lib_x = int(middle_outer_top_landmark.x * image.shape[1])
        middle_outer_top_lib_y = int(middle_outer_top_landmark.y * image.shape[0])
        middle_outer_buttom_lib_x = int(middle_outer_buttom_landmark.x * image.shape[1])
        middle_outer_buttom_lib_y = int(middle_outer_buttom_landmark.y * image.shape[0])
        #
        middle_inner_top_lib_x = int(middle_inner_top_landmark.x * image.shape[1])
        middle_inner_top_lib_y = int(middle_inner_top_landmark.y * image.shape[0])
        middle_inner_buttom_lib_x = int(middle_inner_buttom_landmark.x * image.shape[1])
        middle_inner_buttom_lib_y = int(middle_inner_buttom_landmark.y * image.shape[0])

        right_lip_x = int(right_lip_landmark.x * image.shape[1])
        right_lip_y = int(right_lip_landmark.y * image.shape[0])
        
        max_lip_x = right_lip_x - left_lip_x  # 입술 세로길이
        max_inner_lip_y = middle_inner_buttom_lib_y - middle_inner_top_lib_y  # 안쪽 입술 가로
        max_outer_lip_y = middle_outer_buttom_lib_y - middle_outer_top_lib_y  # 안쪽 입술 가로

       

        print(f"왼쪽 입술 끝 좌표: ({left_lip_x}, {left_lip_y})")
        print(f"오른쪽 입술 끝 좌표: ({right_lip_x}, {right_lip_y})")
        print(f"가운데 inner 윗 입술  좌표: ({middle_inner_top_lib_x}, {middle_inner_top_lib_y})")
        print(f"가운데 inner 아래 입술  좌표: ({middle_inner_buttom_lib_x}, {middle_inner_buttom_lib_y})")
        print(f"가운데 outer 윗 입술  좌표: ({middle_outer_top_lib_x}, {middle_outer_top_lib_y})")
        print(f"가운데 outer 아래 입술  좌표: ({middle_outer_buttom_lib_x}, {middle_outer_buttom_lib_y})")

        print(f"입술 가로길이:({max_lip_x})")
        print(f"입술 inner 세로길이:({max_inner_lip_y})")
        print(f"입술 outer 세로길이:({max_outer_lip_y})")

        if not mb_executed:
            default_lib_x = max_lip_x
            default_lib_inner_y = max_inner_lip_y
            default_lib_outer_y = max_outer_lip_y

            mb_executed = True
        
        
        #입술 길이 82  60  120 75
        #최대길이 127  90  180 110

        # o 5 72
        # u 0

        #얼굴
        cv2.circle(image, (face_top_x, face_top_y), drawing_spec.circle_radius, (0, 0, 255), drawing_spec.thickness)
        cv2.circle(image, (face_buttom_x, face_buttom_y), drawing_spec.circle_radius, (0, 0, 255), drawing_spec.thickness)

        #입
        cv2.circle(image, (middle_outer_top_lib_x, middle_outer_top_lib_y), drawing_spec.circle_radius, (0, 0, 255), drawing_spec.thickness)
        cv2.circle(image, (middle_outer_buttom_lib_x, middle_outer_buttom_lib_y), drawing_spec.circle_radius, (0, 0, 255), drawing_spec.thickness)

        cv2.circle(image, (middle_inner_top_lib_x, middle_inner_top_lib_y), drawing_spec.circle_radius, (0, 0, 255), drawing_spec.thickness)
        cv2.circle(image, (middle_inner_buttom_lib_x, middle_inner_buttom_lib_y), drawing_spec.circle_radius, (255, 0, 255), drawing_spec.thickness)
        cv2.circle(image, (left_lip_x, left_lip_y), drawing_spec.circle_radius, (255, 0, 0), drawing_spec.thickness)
        cv2.circle(image, (right_lip_x, right_lip_y), drawing_spec.circle_radius, (0, 0, 255), drawing_spec.thickness)

        # 텍스트 속성 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        color = (255, 0, 255)  # 보라색


        text ="Smile A"
        # 텍스트 크기 계산
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_width, text_height = text_size

        # 이미지 중앙에 텍스트 위치 계산
        image_center_x = image.shape[1] // 2
        image_center_y = image.shape[0] // 2
        text_x = image_center_x - (text_width // 2)
        text_y = image_center_y + (text_height // 2 )

        # 텍스트 그리기
        print("이거",default_lib_x*2)
        print("이거2",max_lip_x)
        # a
        # if default_lib_x *1.24 < max_lip_x :
        #     cv2.putText(image, "Smile A", (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        # ## e 
        # elif default_lib_x *1.25 < max_lip_x :
        #     cv2.putText(image, "Smile E", (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        # ## i
        # elif default_lib_x *1.25 < max_lip_x :
        #     cv2.putText(image, "Smile I", (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        # ## o
        # elif default_lib_x *1.25 < max_lip_x :
        #     cv2.putText(image, "Smile O", (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
        # ## u
        # elif max_in_lip_y<7 and default_lib_x *0.7 > max_lip_x:
        #     cv2.putText(image, "Smile U", (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    cv2.imshow('MediaPipe Face Mesh - Lips Only (Dots)', image)

    if cv2.waitKey(5) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

import cv2 as cv
import mediapipe as mp

# cap = cv.VideoCapture(0)
img =cv.imread('face.jpeg')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection =mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)
res=face_detection.process(cv.cvtColor(img,cv.COLOR_BGR2RGB))
if not res.detections:
    print("얼굴을 검출할수가 없습니다")
else:
    for detection in res.detections:
        mp_drawing.draw_detection(img,detection)
    cv.imshow("face",img)
cv.waitKey()
cv.destoryAllWindows()
print(mp_face_detection)
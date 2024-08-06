import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)


mp_face_detection = mp.solutions.face_detection

print(mp_face_detection)
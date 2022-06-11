import cv2

HIGH_VALUE = 10000

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, HIGH_VALUE)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HIGH_VALUE)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width, height)

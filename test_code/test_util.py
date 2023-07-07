import cv2
# 얼굴 탐지에만 사용하는 utils
# 왜 얼굴 탐지에만 빼놓았느냐?
# tensorflow 를 import 하면 gpu인지 확인한다고 오래걸림
# 얼굴 탐지쪽은 cascade만 사용하고 tensorflow는 사용하지 않기때문에 분할함

def detect_faces(img, draw_box=True):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	# convert image to grayscale
	# grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# detect faces
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,
		minNeighbors=5,
        minSize=(64, 64),
        flags=cv2.CASCADE_SCALE_IMAGE)
	
    face_box, face_coords = None, []

    for (x, y, w, h) in faces:
        if draw_box:
            cv2.rectangle(img, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 5)
        face_box = img[y-20:y+h+20, x-20:x+w+20]
        face_coords = [x-20,y-20,w+20,h+20]
	# print('img : ', img)
	# print('face_box : ',face_box)
	# print('face_coords : ',face_coords)
    return img, face_box, face_coords

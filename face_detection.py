import cv2
import os 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print(face_cascade)

def gray_scale(img):
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(
		img,
		scaleFactor=1.1,
		minNeighbors=5,
        minSize=(30, 30))
	print(faces)
	print(faces[0])
	print(faces[0][1])
	x, y, w, h = faces[0]
	print(x)
	print(y)
	print(w)
	print(h)
	detect_face_img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)


	cv2.imshow('detect_face img', detect_face_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def detect_faces(img, draw_box=True):
	# convert image to grayscale
	grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# detect faces
	faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.1,
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

# # 원본
# if __name__ == "__main__":
# 	files = os.listdir('sample_faces')
# 	images = [file for file in files if 'jpg' in file]
# 	for image in images:
# 		img = cv2.imread('sample_faces/' + image)
# 		detected_faces, _, _ = detect_faces(img)
# 		cv2.imwrite('sample_faces/detected_faces/' + image, detected_faces)
  
# 		cv2.imshow(image, detected_faces)
# 		cv2.waitKey(0)
# 		cv2.destroyAllWindows()

# K face
if __name__ == "__main__":

	img = cv2.imread('sample_faces/img3.jpg')
	gray_scale(img)


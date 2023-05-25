
import cv2
import numpy as np

face = cv2.imread('rembg.png')

frame = np.ones((face.shape[0], face.shape[1], 3), dtype=np.uint8)

print('face.shape : ', face.shape)
print('frame.shape : ', frame.shape)


add_img2 = cv2.add(face, frame)


cv2.imwrite('./image_add/add.jpg', add_img2)

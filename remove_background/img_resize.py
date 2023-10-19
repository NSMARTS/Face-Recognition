import cv2
from PIL import Image
import numpy as np
###############################################################

img = cv2.imread("myFace1.png")

print("before : ", img.shape)

# resized_img = cv2.resize(img, (106,106,2), interpolation=cv2.INTER_AREA)
resized_img = cv2.resize(img, (256,256,2), interpolation=cv2.INTER_AREA)
print("after : ", img.shape)

##############################################################

# 이미지 읽기
img = Image.open("myFace1.png")

# 이미지 크기 출력
print("Original image size:", img.size)

# 이미지 크기 변경
resized_img = img.resize((106, 106), resample=Image.BICUBIC)

# 변경된 이미지 크기 출력
print("Resized image size:", resized_img.size)

image_array = np.array(resized_img)

print("after resized_img : ", image_array.shape)

cv2.imwrite('aaa.png', image_array)
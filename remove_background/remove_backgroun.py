
## 오브젝트 배경 제거하는 코드

import cv2
# from cvzone.SelfiSegmentationModule import SelfiSegmentation
from PIL import Image
from rembg import remove


# print('imread')
img_imread = cv2.imread('client_face1.png')

# output = Image.open('client_face1.png')
output = remove(img_imread)
# output.save('zzz2.png')

cv2.imwrite('zzz222.png', output)

# cv2.imshow('11', img_imread)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# output = remove(img_imread)


# img_pil = Image.open('removal_background\client_face1.png')

# print('imwrite')
# cv2.imwrite('out.png', output)

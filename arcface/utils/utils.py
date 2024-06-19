# import re
# import os
import time
# import dlib
import cv2
import yaml
import numpy as np
# import pytesseract as tess

from scipy.spatial import distance


# def crop_face_from_id(cv_image, weight_path="weights"):
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(os.path.join(weight_path, 'shape_predictor_68_face_landmarks.dat'))

#     img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

#     face_detector = detector(img_gray, 1)

#     if len(face_detector) == 1:
#         face = face_detector[0]

#         landmarks = predictor(cv_image, face)

#         landmark_list = []
#         for p in landmarks.parts():
#             landmark_list.append([p.x, p.y])
#         h = landmark_list[8][1] - landmark_list[27][1]
#         w = landmark_list[16][0] - landmark_list[0][0]
#         (x, y) = landmark_list[33]

#         cropped = cv_image.copy()
#         cropped = cropped[y - int(h * 1.25): y + h//2, x - int(w * 0.75): x + int(w * 0.75)]

#         return cropped

#     else:
#         raise RuntimeError("[Error] Cannot find ID card")


# def get_idnum(cv_image):
#     """extract birth year from id card by Tesseract"""
#     tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#     config = ("-l kor+eng --oem 3 --psm 4 -c preserve_interword_spaces=1")
#     text = tess.image_to_string(cv_image, config=config)
#     try:
#         text = re.compile(r"\d{6}").search(text).group()
#         # text = text.replace('-', '')
            
#         return text
    
#     except RuntimeError:
#         print("[Error] Cannot find ID card")


# def is_verified_idnum(image):
#     try:
#         id_num = get_idnum(image)

#         weights = np.array([2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5])
#         id_num_array = np.array(list(id_num[:-1]), np.int32)

#         if int(id_num[-1]) == (11 - np.dot(weights, id_num_array.transpose()) % 11) % 11:
#             return True

#         else:
#             return False

#     except RuntimeError:
#         print("[Error] Cannot find ID card")


# def is_verified_age(image):
#     try:
#         id_num = get_idnum(image)
#         year_curr = int(time.strftime('%Y', time.localtime(time.time())))
#         # if id_num[6] == '1' or id_num[6] == '2':
#         if not id_num[0] == '0':
#             year_birth = int(id_num[:2]) + 1900
#         else:
#             year_birth = int(id_num[:2]) + 2000

#         age = year_curr - year_birth
#         if age >= 19:
#             return True
#         else:
#             return False

#     except RuntimeError:
#         print("[Error] Cannot find ID card")


# def eye_aspect_ratio(eye):
#     a = distance.euclidean(eye[1], eye[5])
#     b = distance.euclidean(eye[2], eye[4])
#     c = distance.euclidean(eye[0], eye[3])
#     ear = (a + b) / (2.0 * c)
#     return ear


# def mouth_aspect_ratio(mouth):
#     a = distance.euclidean(mouth[14], mouth[18])    # 51, 59
#     b = distance.euclidean(mouth[4], mouth[8])      # 53, 57
#     c = distance.euclidean(mouth[12], mouth[16])    # 49, 55
#     mar = (a + b) / (2.0 * c)
#     return mar


def l2_norm(x, axis=1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / norm


def get_embeddings(recognition_model, img):
    img = cv2.resize(img, (112, 112))
    img = img / 255.
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    embeds = l2_norm(recognition_model(img))
    return embeds


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def get_ckpt_inf(ckpt_path, steps_per_epoch):
    """get ckpt information"""
    split_list = ckpt_path.split('e_')[-1].split('_b_')
    epochs = int(split_list[0])
    batchs = int(split_list[-1].split('.ckpt')[0])
    steps = (epochs - 1) * steps_per_epoch + batchs

    return epochs, steps + 1

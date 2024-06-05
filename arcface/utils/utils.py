import re
import os
import time
import dlib
import cv2
import yaml
import numpy as np


def crop_face_from_id(cv_image, weight_path="weights"):
    return
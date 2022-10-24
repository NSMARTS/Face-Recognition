name = 'oooo'
import os
import sys
import cv2
# from . import utils_test
import utils_test
from keras.models import load_model
import face_detection
import collections

# validate that the user has ran the model training process
files = os.listdir()
if 'siamese_nn.h5' not in files:
    print("Error: Pre-trained Neural Network not found!")
    print("Please run siamese_nn.py first")
    sys.exit()    

# validate that the user has ran the onboarding process
if 'true_img.png' not in files:
    print("Error: True image not found!")
    print("Please run onbarding.py first")
    sys.exit()    

# load pre-trained Siamese neural network
model = load_model('siamese_nn.h5', custom_objects={'contrastive_loss': utils_test.loss(margin=1), 'euclidean_distance2': utils_test.euclidean_distance})

# prepare the true image obtained during onboard
true_img = cv2.imread('true_img.png', 0)
true_img = true_img.astype('float32')/255
true_img = cv2.resize(true_img, (106, 106))
true_img = true_img.reshape(1, true_img.shape[0], true_img.shape[1], 1)

video_capture = cv2.VideoCapture(0)
preds = collections.deque(maxlen=15)

while True:
    # Capture frame-by-frame
    _, frame = video_capture.read()

    # Detect Faces
    frame, face_img, face_coords = face_detection.detect_faces(frame, draw_box=False)

    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = face_img.astype('float32')/255
        face_img = cv2.resize(face_img, (106, 106))
        face_img = face_img.reshape(1, face_img.shape[0], face_img.shape[1], 1)
        preds.append(1-model.predict([true_img, face_img])[0][0])
        x,y,w,h = face_coords
        if len(preds) == 15 and sum(preds)/15 >= 0.5:
            text = "Identity: {0} \npreds : {1}".format(name, sum(preds)/15)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
        elif len(preds) < 15:
            text = "Identifying ...\npreds : {1}".format(name, sum(preds)/15)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 5)
        else:
            text = "Identity Unknown!\npreds : {1}".format(name, sum(preds)/15)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 5)
        frame = utils_test.write_on_frame(frame, text, face_coords[0], face_coords[1]-10)

    else:
        preds = collections.deque(maxlen=15) # clear existing predictions if no face detected 

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

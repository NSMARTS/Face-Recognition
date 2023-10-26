from keras.models import load_model
import utils_test
import numpy as np

faces_dir = './kface/'
# faces_dir = 'Kface\detect_data/'
# faces_dir = 'att_faces/'

# Import Training and Testing Data
(X_train, Y_train), (X_test, Y_test) = utils_test.get_data(faces_dir)

# create Training Pairs
num_classes = len(np.unique(Y_train))
# training_pairs, training_labels = utils2.create_pairs(X_train, Y_train, num_classes=num_classes)

print('num_classes : ', num_classes)
print('X_train : ', X_train.shape)
print('Y_train : ', Y_train.shape)
training_pairs, training_labels = utils_test.create_pairs(
    X_train, Y_train, num_classes=num_classes)

# exit(1)

# test용
num_classes = len(np.unique(Y_test))
test_pairs, test_labels = utils_test.create_pairs(
    X_test, Y_test, num_classes=num_classes)

test = load_model('siamese_nn_156_0523_1.h5', custom_objects={'contrastive_loss': utils_test.loss(
    margin=1), 'euclidean_distance2': utils_test.euclidean_distance})

predictions = test.predict([test_pairs[:, 0], test_pairs[:, 1]])
utils_test.visualize(training_pairs, training_labels, to_show=20,
                     predictions=predictions, test=True, main_title="predictions")

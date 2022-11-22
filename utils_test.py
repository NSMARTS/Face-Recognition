'''
Helper functions for face recognition
'''
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from keras import backend as K

from tqdm import tqdm

import matplotlib.pyplot as plt


def euclidean_distance(vectors):
    """Find the Euclidean distance between two vectors.
    Arguments:
        vects: List containing two tensors of same length.
    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    # x, y = vectors
    x = vectors[0]
    y = vectors[1]

    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


# def euclidean_distance2(vectors):
#     vector1, vector2 = vectors
#     sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
#     return K.sqrt(K.maximum(sum_square, K.epsilon()))


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.
    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).
    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.
        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.
        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
            # y_true * square_pred + (1-y_true) * margin_square
        )

    return contrastive_loss



def test1(y_true, y_pred):
    return K.mean(y_pred)


def get_data(dir):
    X_train, Y_train = [], []
    X_test, Y_test = [], []
    subfolders = sorted([file.path for file in os.scandir(dir) if file.is_dir()])
    
    print(subfolders);
    print(enumerate(subfolders));
    # foler 순서 : s1->s10->s11...
    for idx, folder in tqdm(enumerate(subfolders)):
        # print(idx)
        # print(folder)
        tmp = 0
        for file in sorted(os.listdir(folder)):
            
            # img = tf.keras.utils.load_img(folder+"/"+file, color_mode='grayscale')
            img = tf.keras.utils.load_img(folder+"/"+file)
            img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)
            #https://linda-suuup.tistory.com/88
            # img = img.resize((120,120))
            
            # print('original img shape : ',img.shape)
            img = cv2.resize(img, (106, 106))
            # img = cv2.resize(img, (128, 128))
            img = tf.keras.utils.img_to_array(img).astype('float32')/255
            img = img.reshape(img.shape[0], img.shape[1],1)
            # img = img.reshape(32,32,1)
            # print(img.shape)
            # print(idx, folder)
            if idx < 120:
                X_train.append(img)
                Y_train.append(idx)
            else:
                X_test.append(img)
                Y_test.append(idx-120)
            tmp = tmp + 1
            # print('---------------------------------- ', tmp)
            # print('resize img.shape : ', img.shape)
            # print('---------------------------------- ')
            # if tmp == 40:
            #     tmp = 0
            #     break
        if idx == 130:
            break

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    print('X_train.shape : ',X_train.shape)
    print('X_test.shape' ,X_test.shape)
    print(Y_train)
    print(Y_test)
    
    return (X_train, Y_train), (X_test, Y_test)


# ********************* 테스트를 위한 테스트 셋, 페어 만들기 *********************
def get_data_test_set(dir):
    X_test, Y_test = [], []
    subfolders = sorted([file.path for file in os.scandir(dir) if file.is_dir()])
    for idx, folder in tqdm(enumerate(subfolders)):

        for file in sorted(os.listdir(folder)):
            
            img = tf.keras.utils.load_img(folder+"/"+file)
            img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)

            img = cv2.resize(img, (106, 106))
            img = tf.keras.utils.img_to_array(img).astype('float32')/255
            img = img.reshape(img.shape[0], img.shape[1],1)

            X_test.append(img)
            Y_test.append(idx)

        if idx == 20:
            break
        # if idx == 100:
        #     break

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_test, Y_test


### pair 개수 줄이기
def create_pairs(X,Y, num_classes):
    pairs, labels = [], []
    print(num_classes)
    # index of images in X and Y for each class
    class_idx = [np.where(Y==i)[0] for i in range(num_classes)]
    # print(num_classes)
    print('np.shape(class_idx) : ',np.shape(class_idx))
    # The minimum number of images across all classes
    min_images = min(len(class_idx[i]) for i in range(num_classes)) - 1
    print('min_image : ',min_images)

    for c in range(num_classes):
        neg_list = list(range(num_classes))
        neg_list.remove(c)
        # print(neg_list)

        neg_idx = 0
        for n in range(min_images):
            # create positive pair
            cnt = 0
            for i in range(n+1, min_images+1):
                cnt += 1
                if(cnt == 5):
                    break
                img1 = X[class_idx[c][random.randint(0, min_images)]]
                img2 = X[class_idx[c][random.randint(0, min_images)]]
                pairs.append((img1, img2))
                labels.append(0)
                
               
                # select a random class from the negative list. 
                # this class will be used to form the negative pair
                neg_c = random.sample(neg_list,1)[0]
                # img1 = X[class_idx[c][n]]
                # img2 = X[class_idx[neg_c][random.sample(range(10), 1)[0]]]

                # 내 image도 골고루 섞이게?
                img1 = X[class_idx[c][neg_idx]]
                img2 = X[class_idx[neg_c][random.randint(0, min_images)]]
                pairs.append((img1,img2))
                labels.append(1)
                
                neg_idx = neg_idx+1
                neg_idx = neg_idx % (min_images+1)
##################################################################################               
    print()
    print('시험 데이터의 개수 : ',len(pairs))
    print('정답 데이터의 개수 : ',np.shape(labels))
    print()
    return np.array(pairs), np.array(labels).astype("float32")
    
"""
## Visualize pairs and their labels
"""

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False, main_title="Figure"):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.
    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).
    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images    
    fig, axes = plt.subplots(num_row, num_col, figsize=(10, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    # if test:
    #     plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    # else:
    #     plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    fig.suptitle(main_title)
    plt.show()




def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.
    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.
    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()




def write_on_frame(frame, text, text_x, text_y):
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
    box_coords = ((text_x, text_y), (text_x+text_width+20, text_y-text_height-20))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    return frame
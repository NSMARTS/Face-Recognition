'''
Main code for training a Siamese neural network for face recognition
https://keras.io/examples/vision/siamese_contrastive/

TF version2

'''
from this import s
# from . import utils_test
import utils_test
import numpy as np
import keras
import tensorflow as tf
from keras import optimizers
from keras import layers


from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D


import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#     try:
#         print('================')
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)

# os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')

epochs = 100
batch_size = 4
margin = 1

faces_dir = './kface/'
# faces_dir = 'Kface\detect_data/'
# faces_dir = 'att_faces/'

# Import Training and Testing Data
(X_train, Y_train), (X_test, Y_test) = utils_test.get_data(faces_dir)


# print(X_train.shape)
# print(X_train[0])

# image = X_test[0]
# # plot the sample
# fig = plt.figure
# plt.imshow(image, cmap='gray')
# plt.show()
# exit(1)


# create Training Pairs
num_classes = len(np.unique(Y_train))
# training_pairs, training_labels = utils2.create_pairs(X_train, Y_train, num_classes=num_classes)

print('num_classes : ',num_classes)
print('X_train : ',X_train.shape)
print('Y_train : ', Y_train.shape)
training_pairs, training_labels = utils_test.create_pairs(X_train, Y_train, num_classes=num_classes)

# exit(1)

# test용
num_classes = len(np.unique(Y_test))
test_pairs, test_labels = utils_test.create_pairs(X_test, Y_test, num_classes=num_classes)


"""
이미지 pair 보기
"""
# Inspect training pairs
# utils_test.visualize(training_pairs[:-1], training_labels[:-1], to_show=12, num_col=4, main_title='Training pairs sample')


# # Inspect validation pairs
# utils2.visualize(pairs_val[:-1], labels_val[:-1], to_show=12, num_col=4, 'validation pairs sample')

# Inspect test pairs
# utils2.visualize(test_pairs[:-1], test_labels[:-1], to_show=12, num_col=4, main_title='Test pairs sample')

'''

Siamese Model 설정

'''



# model = Sequential(name='Shared_Conv_Network')
# input_shape = X_train.shape[1:]
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=input_shape))
# model.add(MaxPooling2D())
# model.add(Dropout(0.1))  
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))    
# model.add(Dropout(0.1))  
# model.add(Flatten())
# model.add(Dense(units=128, activation='sigmoid'))

# embedding_network = model


# https://datascience.stackexchange.com/questions/82486/siamese-network-sigmoid-function-to-compute-similarity-score
model = Sequential(name='Shared_Conv_Network')
input_shape = X_train.shape[1:]
print(input_shape)
model.add(Conv2D(filters=32,  padding="same", kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D())
model.add(Dropout(0.1))  
model.add(Conv2D(filters=64,  padding="same", kernel_size=(3,3), activation='relu'))    
model.add(MaxPooling2D())
model.add(Conv2D(filters=128,  padding="same", kernel_size=(3,3), activation='relu'))    
model.add(MaxPooling2D())
model.add(Dropout(0.1))  
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(units=10, activation='relu'))

embedding_network = model



# input = layers.Input(X_train.shape[1:])
# x = layers.Conv2D(64, (3, 3), activation="relu")(input)
# x = layers.MaxPooling2D()(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Conv2D(32, (3, 3), activation="relu")(x)
# x = layers.MaxPooling2D()(x)
# x = layers.Dropout(0.1)(x)
# x = layers.Flatten()(x)
# x = layers.Dense(128, activation="sigmoid")(x)

# input = layers.Input(X_train.shape[1:])
# # x = layers.BatchNormalization()(input)
# x = layers.Conv2D(64, (3, 3), activation="relu")(input)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# # x = layers.Dropout(0.1)(x)
# x = layers.Conv2D(32, (3, 3), activation="relu")(x)
# x = layers.MaxPooling2D(pool_size=(2, 2))(x)
# x = layers.Flatten()(x)

# # x = layers.BatchNormalization()(x)
# x = layers.Dense(128, activation="sigmoid")(x)

# embedding_network = keras.Model(input, x)




input_1 = layers.Input(X_train.shape[1:])
input_2 = layers.Input(X_train.shape[1:])

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(utils_test.euclidean_distance)([tower_1, tower_2])
model = keras.Model(inputs=[input_1, input_2], outputs=merge_layer)

# normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
# output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
# model = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

# adam = optimizers.Adam(lr=0.00005)
# adam = optimizers.adam_v2(lr=0.00005)

# opt = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
opt = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
model.compile(loss=utils_test.loss(margin=margin), optimizer=opt, metrics=["accuracy"])

# model.compile(loss=utils2.loss(margin=margin), optimizer='adam', metrics=["accuracy"])
model.summary()

history = model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
                    validation_split=0.2,
                    batch_size=batch_size,
                    epochs=epochs)


"""
## Visualize results
"""
# Plot the accuracy
utils_test.plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the constrastive loss
utils_test.plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

"""
## Evaluate the model
"""
# results = model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
# print("test loss, test acc:", results)

"""
## Visualize the predictions
"""
# predictions = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
# utils_test.visualize(test_pairs, test_labels, to_show=12, predictions=predictions, test=True, main_title = "predictions")


# Save the model
model.save('siamese_nn.h5')

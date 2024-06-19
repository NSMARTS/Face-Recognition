import tensorflow as tf

if tf.config.list_physical_devices('GPU'):

  print('GPU is available.')

else:

  print('GPU is NOT available. Make sure TensorFlow version less than 2.11 and Installed all GPU drivers.')

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True 

session = tf.compat.v1.Session(config=config)
# Python 3.8.6
import time

# tensorflow 2.4.0
# matplotlib 3.3.3
# numpy 1.19.4
# opencv-python 4.4.0
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.datasets as datasets
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt
import numpy as np
import cv2

fout = open('test2_wei_firstnn.txt', 'w')
now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)
print("[TIMER] Process Time:", now, file = fout, flush = True)

# File location to save to or load from
MODEL_SAVE_PATH = './cifar_net.pth'
# Set to zero to use above saved model
TRAIN_EPOCHS = 200
# If you want to save the model at every epoch in a subfolder set to 'True'
SAVE_EPOCHS = False
# If you just want to save the final output in current folder, set to 'True'
SAVE_LAST = False
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_TEST = 4

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print('[INFO] GPU is detected.')
    print('[INFO] GPU is detected.', file = fout, flush = True)
else:
    print('[INFO] GPU not detected.')
    print('[INFO] GPU not detected.', file = fout, flush = True)
print('[INFO] Done importing packages.')
print('[INFO] Done importing packages.', file = fout, flush = True)

class Net():
    def __init__(self, input_shape):
        # input_shape is assumed to be 4 dimensions: 1. Batch Size, 2. Image Width, 3. Image Height, 4. Number of Channels.
        # You might see this called "channels_last" format.
        self.model = models.Sequential()
        # For the first convolution, you need to give it the input_shape.  Notice that we chop off the batch size in the function.
        # In our example, input_shape is 4 x 32 x 32 x 3.  But then it becomes 32 x 32 x 3, since we've chopped off the batch size.
        # For Conv2D, you give it: Outgoing Layers, Frame size.  Everything else needs a keyword.
        # Popular keyword choices: strides (default is strides=1), padding (="valid" means 0, ="same" means whatever gives same output width/height as input).  Not sure yet what to do if you want some other padding.
        # Activation function is built right into the Conv2D function as a keyword argument.
        # self.model.add(layers.Conv2D(6, 5, input_shape = input_shape, activation = 'relu'))
        # # In our example, output from first Conv2D is 28 x 28 x 6.
        # # For MaxPooling2D, default strides is equal to pool_size.  Batch and layers are assumed to match whatever comes in.
        # self.model.add(layers.MaxPooling2D(pool_size = 2))
        # # In our example, we are now at 14 x 14 x 6.
        # self.model.add(layers.Conv2D(16, 5, activation = 'relu'))
        # # In our example, we are now at 10 x 10 x 16.
        # self.model.add(layers.MaxPooling2D(pool_size = 2))
        # # In our example, we are now at 5 x 5 x 16.
        # self.model.add(layers.Flatten())
        # # Now, we flatten to one dimension, so we go to just length 400.
        # self.model.add(layers.Dense(120, activation = 'relu'))
        # # Now we're at length 120.
        # self.model.add(layers.Dense(84, activation = 'relu'))
        # # Now we're at length 84.
        # self.model.add(layers.Dense(10))
        # Now we're at length 10, which is our number of classes.

        self.model.add(layers.Conv2D(6, 7, input_shape = input_shape, activation = 'relu'))
        self.model.add(layers.MaxPooling2D(pool_size = 2))
        self.model.add(layers.Conv2D(12, 5, activation = 'relu'))
        self.model.add(layers.MaxPooling2D(pool_size = 2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(120, activation = 'relu'))
        self.model.add(layers.Dense(84, activation = 'relu'))
        self.model.add(layers.Dense(10))

        self.optimizer = optimizers.SGD(lr=0.001, momentum=0.9)
        self.loss = losses.MeanSquaredError()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def __str__(self):
        self.model.summary(print_fn = self.print_summary)
        return ""

    def print_summary(self, summaryStr):
        print(summaryStr)
        print(summaryStr, file=fout)

print("[INFO] Loading Traning and Test Datasets.")
print("[INFO] Loading Traning and Test Datasets.", file=fout)

# Get the CIFAR-10 Dataset.
((trainX, trainY), (testX, testY)) = datasets.cifar10.load_data()
# Convert from integers 0-255 to decimals 0-1.
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Convert labels from integers to vectors.
lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

net = Net((32, 32, 3))
# Notice that this will print both to console and to file.
print(net)

results = net.model.fit(trainX, trainY, validation_data=(testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1)

plt.figure()
plt.plot(np.arange(0, 200), results.history['loss'])
plt.plot(np.arange(0, 200), results.history['val_loss'])
plt.plot(np.arange(0, 200), results.history['accuracy'])
plt.plot(np.arange(0, 200), results.history['val_accuracy'])
plt.show()

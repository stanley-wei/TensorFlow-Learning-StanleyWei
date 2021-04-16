import time
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

fout = open('results/test_reuters.txt', 'w')
now = time.strftime("%H:%M:%S", time.localtime())
print("[TIMER] Process Time:", now)
print("[TIMER] Process Time:", now, file = fout, flush = True)

MODEL_SAVE_PATH = './reuters_net.pth'
TRAIN_EPOCHS = 200
SAVE_EPOCHS = False
SAVE_LAST = True
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
        self.model = models.Sequential()

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

((trainX, trainY), (testX, testY)) = datasets.cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

net = Net((32, 32, 3))

print(net)

results = net.model.fit(trainX, trainY, validation_data=(testX, testY), shuffle = True, epochs = TRAIN_EPOCHS, batch_size = BATCH_SIZE_TRAIN, validation_batch_size = BATCH_SIZE_TEST, verbose = 1)

plt.figure()
plt.plot(np.arange(0, 200), results.history['loss'])
plt.plot(np.arange(0, 200), results.history['val_loss'])
plt.plot(np.arange(0, 200), results.history['accuracy'])
plt.plot(np.arange(0, 200), results.history['val_accuracy'])
plt.show()

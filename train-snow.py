import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)'

from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import preprocess_input

import os
import glob
import cv2

from tqdm import tqdm


resnet_weights_path = 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

basemodel = Sequential()
basemodel.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))


basemodel.layers[0].trainable = False

def extract_vector(path):
    resnet_feature_list = []

    for im in tqdm(sorted(glob.glob(path))):

        im = cv2.imread(im)
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = basemodel.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())

    return np.array(resnet_feature_list)


if 'data' in os.listdir('.'):
    data = np.genfromtxt('data', delimiter=' ')
else: 
    data = extract_vector('augmented-images/*')
    np.savetxt('data', data)

if 'data2' in os.listdir('.'):
    data2 = np.genfromtxt('data2', delimiter=' ')
else:
    data2 = extract_vector('training-images/*')
    np.savetxt('data2', data2)

labels = pd.read_excel('training-labels.xlsx')

y = np.repeat(labels.snow, 5).values

y = np.vstack((y.reshape(-1,1), labels.snow.values.reshape(-1,1))).reshape(-1,)

data = np.vstack((data, data2))

model = Sequential()
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(data, y, batch_size=128, epochs=250, validation_split=0.2, class_weight={0: 0.1, 1: 1})
model.save("model-snow.h5")

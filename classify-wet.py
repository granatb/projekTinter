import numpy as np
import pandas as pd
from keras.applications import ResNet50
from keras.models import Sequential, load_model
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

model = load_model('model-wet.h5')

if 'data-to-classify' in os.listdir('.'):
    data_to_classify = np.genfromtxt('data-to-classify', delimiter=' ')
else: 
    data_to_classify = extract_vector('images-to-classify/*')
    np.savetxt('data-to-classify', data_to_classify)

pd.DataFrame(list(zip(sorted(glob.glob('images-to-classify/*')),*list(zip(*model.predict_classes(data_to_classify))))), columns=["image","wet"]).to_csv("classification-wet.csv",index=False)
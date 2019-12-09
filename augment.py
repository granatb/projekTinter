# importing necessary functions 
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img, array_to_img
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import re

from tqdm import tqdm

from glob import glob
# images stored in the trainings folder
img_list = sorted(glob("./training-images/*"))

ia.seed(1)

# transformation that will be applied to the image
seq = iaa.Sequential([
        #iaa.Fliplr(0.5), # horizontal flips
        #iaa.Crop(percent=(0, 0.1)), # random crops
    
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.25)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.

        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.90, 1.10), per_channel=0.2),
    ], random_order=True) # apply augmenters in random order

    
    
for i, im in enumerate(tqdm(sorted(glob("./training-images/*")))):
    # load an image
    img = load_img(im)  
    
    # convert an image to an array
    img_arr = img_to_array(img) 
    
    # multiply an array (image) by 5
    images = np.array(
        [img_arr for _ in range(5)],
        dtype=np.uint8
    )
    
    # apply the transformation to each array
    images_aug = seq(images=images)
    
    for j in range(5):
        # convert an array to an image
        img_aug = array_to_img(images_aug[j])
        
        # save a new image
        save_img(path = "./augmented-images/" + str(i) + "_" + str(j) + ".png" ,
                 x = img_aug,
                 file_format='png')
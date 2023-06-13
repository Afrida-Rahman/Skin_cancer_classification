import os

import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from skimage import io

os.chdir("..")


def aug(type):
    ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    for j in ctg:
        image_directory = f"data/raw_data/72_8_20/224/{type}/{j}/"
        aug_directory = f"data/keras_aug_data/72_8_20/224/{type}/{j}/"
        SIZE = 224
        dataset = []

        datagen = ImageDataGenerator(
            rotation_range=80,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=True,
            featurewise_std_normalization=True,
            featurewise_center=True,
            zca_whitening=True,
            brightness_range=(0.5, 1.5))
        my_images = os.listdir(image_directory)
        for i, image_name in enumerate(my_images):
            if image_name.split('.')[1] == 'jpg':
                image = io.imread(image_directory + image_name)
                image = Image.fromarray(image, 'RGB')
                image = image.resize((SIZE, SIZE))
                dataset.append(np.array(image))
        x = np.array(dataset)
        i = 0
        if not os.path.exists(aug_directory):
            os.makedirs(aug_directory)
        datagen.flow(x, seed=197, batch_size=16, save_to_dir=aug_directory, save_prefix='dr', save_format='jpg')
        # i += 1
        # if i > 100:
        #     break

        print(f"{j} is augmented")


aug("train")

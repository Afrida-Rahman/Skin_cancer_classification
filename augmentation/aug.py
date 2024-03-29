import os
from glob import glob

import imageio.v3 as iio
import imgaug.augmenters as iaa
from PIL import Image

os.chdir("..")


def augmentation(filepath, aug_path):
    input_img = iio.imread(filepath)
    img = filepath.split('/')[-1].split('.')[0]

    # save original file

    input_img_array = Image.fromarray(input_img)
    input_img_array.save(aug_path + img + '.jpg')

    input_hf = Image.fromarray(iaa.Fliplr(p=1).augment_image(input_img))
    input_hf.save(aug_path + img + '_hf.jpg')

    input_vf = Image.fromarray(iaa.Flipud(p=1).augment_image(input_img))
    input_vf.save(aug_path + img + '_vf.jpg')

    input_n_rot = Image.fromarray(
        iaa.Affine(rotate=-10, fit_output=True).augment_image(input_img))
    input_n_rot.save(aug_path + img + '_nr.jpg')

    input_p_rot = Image.fromarray(iaa.Affine(rotate=10, fit_output=True).augment_image(input_img))
    input_p_rot.save(aug_path + img + '_pr.jpg')

    input_contrast = Image.fromarray(iaa.GammaContrast(gamma=(.3, 3.2)).augment_image(input_img))
    input_contrast.save(aug_path + img + '_gc.jpg')

    input_bright = Image.fromarray(iaa.AddToBrightness().augment_image(input_img))
    input_bright.save(aug_path + img + '_br.jpg')

    input_bc = iaa.GammaContrast().augment_image(input_img)
    input_bc = Image.fromarray(iaa.AddToBrightness().augment_image(input_bc))
    input_bc.save(aug_path + img + '_bc.jpg')

    # crop1 = iaa.Crop(percent=(0.1, 0.3))
    # input_crop1 = crop1.augment_image(input_img)
    # input_crop1 = Image.fromarray(input_crop1)
    # input_crop1.save(aug_path + img + '_crp1.jpg')

    # crop2 = iaa.Crop(percent=(0.2, 0))
    # input_crop2 = crop2.augment_image(input_img)
    # input_crop2 = Image.fromarray(input_crop2)
    # input_crop2.save(aug_path + img + '_crp2.jpg')

    # jitter_img = Image.fromarray(input_img)
    # jitter_img = ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)).forward(jitter_img)
    # jitter_img.save(aug_path + img + '_jitter.jpg')


file_path = f'data/raw_data/class_separated_data/'
aug_path = f'data/sensor_data/augmented_class/'

ctg = ['akiec', 'bcc', 'df', 'vasc']

for i in ctg:
    if not os.path.exists(aug_path + i):
        os.makedirs(aug_path + i)

for i in ctg:
    files = glob(file_path + i + '/*')
    for j in files:
        # print(j)
        augmentation(j, aug_path + i + '/')
    print(f"{i} is augmented")

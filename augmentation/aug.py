import os

import imageio.v3 as iio
import imgaug.augmenters as iaa
from glob import glob
from PIL import Image


def augmentation(filepath, aug_path):
    input_img = iio.imread(filepath)
    input_hf = Image.fromarray(iaa.Fliplr(p=1.0).augment_image(input_img))
    input_hf.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_hf.jpg')

    input_vf = Image.fromarray(iaa.Flipud(p=1.0).augment_image(input_img))
    input_vf.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_vf.jpg')

    input_n_rot = Image.fromarray(iaa.Affine(rotate=-10, fit_output=True).augment_image(input_img))
    input_n_rot.save(aug_path + filepath.split('/')[3].split('.')[0] + '_nr.jpg')

    input_p_rot = Image.fromarray(iaa.Affine(rotate=10, fit_output=True).augment_image(input_img))
    input_p_rot.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_pr.jpg')

    input_contrast = Image.fromarray(iaa.GammaContrast(gamma=(.3, 3.2)).augment_image(input_img))
    input_contrast.save(aug_path + filepath.split('/')[3].split('.')[0] + '_gc.jpg')

    input_bright = Image.fromarray(iaa.AddToBrightness().augment_image(input_img))
    input_bright.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_br.jpg')

    input_bc = iaa.GammaContrast().augment_image(input_img)
    input_bc = Image.fromarray(iaa.AddToBrightness().augment_image(input_bc))
    input_bc.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_bc.jpg')


file_path = 'raw_data/class_separated_data/'
aug_path = 'aug_data/'
ctg = ['akiec', 'bcc', 'df', 'vasc']

for i in ctg:
    os.makedirs(aug_path+i)

for i in ctg:
    files = glob(file_path+i+ '/*')
    for j in files:
        augmentation(j , aug_path+i+'/')
    print(f"{i} is augmented")

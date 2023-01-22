import os

import imageio.v3 as iio
import imgaug.augmenters as iaa
from glob import glob
from PIL import Image
from torchvision.transforms import ColorJitter


def augmentation(filepath, aug_path):
    input_img = iio.imread(filepath)
    input_hf = Image.fromarray(iaa.Fliplr(p=1.0).augment_image(input_img))
    input_hf.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_hf.jpg')

    input_vf = Image.fromarray(iaa.Flipud(p=1.0).augment_image(input_img))
    input_vf.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_vf.jpg')

    input_n_rot = Image.fromarray(iaa.Affine(rotate=-20, fit_output=True).augment_image(input_img))
    input_n_rot.save(aug_path + filepath.split('/')[3].split('.')[0] + '_nr.jpg')

    input_p_rot = Image.fromarray(iaa.Affine(rotate=20, fit_output=True).augment_image(input_img))
    input_p_rot.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_pr.jpg')
    #
    # input_contrast = Image.fromarray(iaa.GammaContrast(gamma=(.3, 3.2)).augment_image(input_img))
    # input_contrast.save(aug_path + filepath.split('/')[3].split('.')[0] + '_gc.jpg')
    #
    # input_bright = Image.fromarray(iaa.AddToBrightness().augment_image(input_img))
    # input_bright.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_br.jpg')
    #
    # input_bc = iaa.GammaContrast().augment_image(input_img)
    # input_bc = Image.fromarray(iaa.AddToBrightness().augment_image(input_bc))
    # input_bc.save(aug_path+ filepath.split('/')[3].split('.')[0] + '_bc.jpg')
    #
    crop1 = iaa.Crop(percent=(0, 0.933))
    input_crop1 = crop1.augment_image(input_img)
    input_crop1.save(aug_path + filepath.split('/')[3].split('.')[0] + '_crp1.jpg')

    crop2 = iaa.Crop(percent=(0.933, 0))
    input_crop2 = crop2.augment_image(input_img)
    input_crop2 = Image.fromarray(input_crop2)
    input_crop2.save(aug_path + filepath.split('/')[3].split('.')[0] + '_crp2.jpg')

    jitter_img = Image.fromarray(input_img)
    jitter_img = ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)).forward(jitter_img)
    jitter_img.save(aug_path + filepath.split('/')[3].split('.')[0] + '_jitter.jpg')

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

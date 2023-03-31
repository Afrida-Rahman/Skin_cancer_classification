import os
from glob import glob

import imageio.v3 as iio
from PIL import Image
from torchvision.transforms import RandomHorizontalFlip, \
    RandomVerticalFlip, RandomAdjustSharpness, RandomRotation, RandomCrop, RandomAutocontrast, RandomEqualize, \
    RandomInvert, RandomGrayscale

os.chdir("..")


def augmentation(filepath, aug_path):
    input_img = iio.imread(filepath)
    input_img = Image.fromarray(input_img)
    img = filepath.split('/')[-1].split('.')[0]

    # save original file

    # input_img_array = Image.fromarray(input_img)
    input_img.save(aug_path + img + '.jpg')

    input_hf = RandomHorizontalFlip(0.5).forward(input_img)
    input_hf.save(aug_path + img + '_hf.jpg')

    input_vf = RandomVerticalFlip(0.5).forward(input_img)
    input_vf.save(aug_path + img + '_vf.jpg')

    input_rot_n = RandomRotation(degrees=(-20, 0)).forward(input_img)
    input_rot_n.save(aug_path + img + '_rot_n.jpg')

    input_rot_p = RandomRotation(degrees=(0, 20)).forward(input_img)
    input_rot_p.save(aug_path + img + '_rot_p.jpg')

    input_contrast = RandomAutocontrast(0.5).forward(input_img)
    input_contrast.save(aug_path + img + '_contrast.jpg')

    input_sharpness = RandomAdjustSharpness(sharpness_factor=6, p=0.5).forward(input_img)
    input_sharpness.save(aug_path + img + '_sharpness.jpg')

    input_grayscale = RandomGrayscale(0.5).forward(input_img)
    input_grayscale.save(aug_path + img + '_grayscale.jpg')

    input_crop = RandomCrop(size=384, pad_if_needed=True).forward(input_img)
    input_crop.save(aug_path + img + '_crp.jpg')

    # input_invert = RandomInvert(0.5).forward(input_img)
    # input_invert.save(aug_path + img + '_invert.jpg')

    # input_posterize = RandomPosterize(bits=7, p=0.5).forward(input_img)
    # input_posterize.save(aug_path + img + '_post.jpg')

    # input_equalize = RandomEqualize(0.5).forward(input_img)
    # input_equalize.save(aug_path + img + '_eq.jpg')

    # jitter_img = Image.fromarray(input_img)
    # jitter_img = ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2)).forward(jitter_img)
    # jitter_img.save(aug_path + img + '_jitter.jpg')


file_path = f'data/raw_data/class_separated_data/'
aug_path = f'data/sensor_data/pytorch/aug_imbalanced/'

ctg = ['akiec', 'bcc', 'df', 'vasc']

for i in ctg:
    if not os.path.exists(aug_path + i):
        os.makedirs(aug_path + i)

for i in ctg:
    files = glob(file_path + i + '/*')
    # print(files)
    for j in files:
        # print(j)
        augmentation(j, aug_path + i + '/')
    print(f"{i} is augmented")

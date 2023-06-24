import os
from glob import glob

import numpy as np
import pandas as pd
import splitfolders
import torch
import torchvision.transforms as fn
import torchvision.transforms as transforms
from PIL import Image

os.chdir("..")
resolution = 384
input_file_path = "data/raw_data/class_separated_data/"
output_file_path = "data/raw_data/92_8/384_norm/"


def split_train_test_val(ratio):
    splitfolders.ratio(input=input_file_path,
                       output=output_file_path,
                       seed=2975, ratio=ratio, group_prefix=None)


def resize_img(file_path, resolution):
    folders = glob(file_path + '/*')
    for i in folders:
        image = Image.open(i)
        image = image.resize((resolution, resolution))
        tensor_img = fn.ToTensor()(image)
        norm_img = fn.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(tensor_img)
        img = fn.ToPILImage()(norm_img)
        os.remove(file_path + '/' + i.split('/')[-1])
        img.save(file_path + '/' + i.split('/')[-1])


def separate_class_label(file_path, ctg):
    folders = glob(file_path + ctg + '/*')
    img_list, label_list = [], []
    for i in folders:
        image = Image.open(i)
        # image = cv2.imread(i)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = Image.fromarray(image)
        image = image.resize((28, 28))
        img_array = np.asarray(image)
        norm_array = img_array / 255
        # norm_array = np.expand_dims(norm_array, axis=-1)
        img_list.append(norm_array)
        label_list.append(i.split('/')[3])

    return img_list, label_list


def separate_class_label_count_wise(file_path, ctg, count):
    folders = glob(file_path + ctg + '/*')
    img_list, label_list = [], []
    for idx, i in enumerate(folders):
        if idx < count:
            image = Image.open(i)
            image = image.resize((72, 72))
            img_array = np.asarray(image)
            norm_array = img_array / 255
            img_list.append(norm_array)
            label_list.append(i.split('/')[3])
        else:
            break

    return img_list, label_list


def convert_label_to_int(label_list):
    a = []
    for i in label_list:
        if i == "akiec":
            a.append(0)
        if i == "bcc":
            a.append(1)
        if i == "bkl":
            a.append(2)
        if i == "df":
            a.append(3)
        if i == "mel":
            a.append(4)
        if i == "vasc":
            a.append(5)
        if i == "nv":
            a.append(6)
    return np.array(a)


def prepare_dataset_labels():
    df = pd.read_csv("HAM10000_metadata")
    path = "dataset/HAM10000_images/"
    images = df["image_id"].tolist()
    label = df["dx"].tolist()
    for i in range(len(label)):
        image = Image.open(path + images[i] + ".jpg")
        image.save("dataset/" + label[i] + "/" + images[i] + ".jpg")
        print("dataset/" + label[i] + "/" + images[i])


def normalize_img(file_path):
    folders = glob(file_path + '/*')
    for i in folders:
        img = Image.open(i)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tr = transform(img)
        mean, std = img_tr.mean([1, 2]), img_tr.std([1, 2])

        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        img_nor = transform_norm(img)
        img_nor = img_nor.numpy().astype(np.float16)
        img_nor = torch.Tensor(img_nor)
        transform = transforms.ToPILImage()
        img_nor = transform(img_nor)
        os.remove(file_path + '/' + i.split('/')[-1])
        img_nor.save(file_path + '/' + i.split('/')[-1])


###  SPLIT ###
# split_train_test_val(ratio=(.9173, .0827))

# RESIZE #
categories = ["akiec", "bcc", "bkl", "df", "mel", "vasc", "nv"]
for i in categories:
    # resize_img(file_path=output_file_path + 'train/' + i, resolution=resolution)
    # resize_img(file_path=output_file_path + 'val/' + i, resolution=resolution)
    # resize_img(file_path=output_file_path + 'test/' + i, resolution=resolution)
    normalize_img(file_path=output_file_path + 'train/' + i)
    normalize_img(file_path=output_file_path + 'val/' + i)
    print(f"{i} is done")

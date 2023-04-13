import os
from glob import glob

import numpy as np
import pandas as pd
import splitfolders
from PIL import Image

os.chdir("..")
input_file_path = "data/raw_data/class_separated_data/"
output_file_path = "data/raw_data/72_8_20/384/"


def split_train_test_val(ratio):
    splitfolders.ratio(input=input_file_path,
                       output=output_file_path,
                       seed=201, ratio=ratio, group_prefix=None)


def resize_img(file_path, resolution):
    folders = glob(file_path + '/*')
    for i in folders:
        image = Image.open(i)
        image = image.resize((resolution, resolution))
        os.remove(file_path + '/' + i.split('/')[-1])
        image.save(file_path + '/' + i.split('/')[-1])


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


###  SPLIT ###
split_train_test_val(ratio=(.72, .08, .20))

# RESIZE #
categories = ["akiec", "bcc", "bkl", "df", "mel", "vasc", "nv"]
for i in categories:
    resize_img(file_path=output_file_path + 'train/' + i, resolution=384)
    resize_img(file_path=output_file_path + 'val/' + i, resolution=384)
    resize_img(file_path=output_file_path + 'test/' + i, resolution=384)
    print(f"{i} is done")

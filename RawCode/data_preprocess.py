import os

import cv2
from PIL import Image
from glob import glob
import splitfolders
import numpy as np
from tensorflow.keras import utils

os.chdir("..")


def split_train_test_val():
    splitfolders.ratio(input="aug_data/balanced/class_separated_data",
                       output="aug_data/balanced/train_test_val",
                       seed=100, ratio=(.7, .2, .1), group_prefix=None)


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


####  SPLIT ###
split_train_test_val()

### TRAIN #####
# train_path = "../raw_data/train_test_splitted/train/"
#
# label_list = ["akiec", "bcc", "bkl", "df", "mel", "vasc", "nv"]
# train_img, train_label = [], []
# for i in label_list:
#     train_img1, train_label1 = separate_class_label(train_path, i)
#     print(f"{i} done")
#     train_img.extend(train_img1)
#     train_label.extend(train_label1)
#
# train_label = np.array(train_label)
# train_img = np.array(train_img)
#
# print(train_img.shape)
# print(train_label.shape)
# print(f"x_train: {train_img.shape} - y_train: {train_label.shape}")
#
# train_label = convert_label_to_int(train_label)
# train_Y_ctg= utils.to_categorical(train_label)
# np.save(arr=train_Y_ctg, file="../random_trial/train_label_ctg_3.npy")
#
# print("train conversion is done")
#
# np.save(arr=train_img, file="../random_trial/train_img_28x28x3.npy")
# np.save(arr=train_label, file="random_trial/train_label_28x28x3.npy")
# print("train saved successfully.")
#
# #### TEST #####
# test_path = "../raw_data/train_test_splitted/val/"
#
# label_list = ["akiec", "bcc", "bkl", "df", "mel", "vasc", "nv"]
# test_img, test_label = [], []
# for i in label_list:
#     train_img1, train_label1 = separate_class_label(test_path, i)
#     print(f"{i} done")
#     test_img.extend(train_img1)
#     test_label.extend(train_label1)
#
# test_label = np.array(test_label)
# test_img = np.array(test_img)
# print(test_img.shape)
# print(test_label.shape)
# print(f"x_test: {test_img.shape} - y_test: {test_label.shape}")
#
# test_label = convert_label_to_int(test_label)
# test_Y_ctg = utils.to_categorical(test_label)
# np.save(arr=test_Y_ctg, file="../random_trial/test_label_ctg_3.npy")
# print("test conversion is done")
#
# np.save(arr=test_img, file="../random_trial/test_img_28x28x3.npy")
# np.save(arr=test_label, file="random_trial/test_label_28x28x3.npy")
# print("test saved successfully.")

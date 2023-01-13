import os
from glob import glob
from PIL import Image
import random


os.chdir("..")
file_path = "aug_data/imbalanced/class_separated_data/"
result_path = "aug_data/balanced/class_separated_data/"

ctg = ['akiec','bcc','vasc']
selection_num = 1099


def select(ctg):
    files = glob(file_path + ctg + '/*')
    random.shuffle(files)
    # print(f"{ctg} = {files[:2]}")
    j = 0
    for i in files:
        if j < selection_num:
            j += 1
            image = Image.open(i)
            print(i)
            image.save(result_path + ctg + '/' + i.split('/')[4])
        else:
            break


for i in ctg:
    os.makedirs(result_path + i)
    select(i)
    print(f"{i} selection is done!")

import os
from glob import glob
from PIL import Image
import random


os.chdir("..")
file_path = "data/aug_data/data_85_15_split/imbalanced/train/"
result_path = "data/aug_data/data_85_15_split/balanced/train/"

ctg = ['akiec','bcc']


def select(ctg):
    selection_num = 1000 - len(glob(result_path + ctg + '/*'))
    files = glob(file_path + ctg + '/*')
    random.shuffle(files)
    # print(f"{ctg} = {files[:2]}")
    j = 0
    for i in files:
        if j < selection_num:
            j += 1
            image = Image.open(i)
            print(i)
            image.save(result_path + ctg + '/' + i.split('/')[6])
        else:
            break


for i in ctg:
    # os.makedirs(result_path + i)
    select(i)
    print(f"{i} selection is done!")

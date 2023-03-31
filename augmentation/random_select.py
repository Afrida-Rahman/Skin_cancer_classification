import os
from glob import glob

from PIL import Image
from numpy import random

os.chdir("..")
file_path = f'data/sensor_data/pytorch/aug_imbalanced/'
result_path = f'data/sensor_data/pytorch/aug_balanced/'
# raw_data_path = f"data/raw_data/data_{d_type}_split/train/"

ctg = ['akiec', 'bcc', 'df', 'vasc']


# copy_tree(raw_data_path, result_path)


def select(ctg):
    selection_num = 1099  # - len(glob(result_path + ctg + '/*'))
    files = glob(file_path + ctg + '/*')
    random.shuffle(files)
    # print(f"{ctg} = {files[:2]}")
    j = 0
    for i in files:
        if j < selection_num:
            j += 1
            image = Image.open(i)
            # print(i)
            image.save(result_path + ctg + '/' + i.split('/')[-1])
        else:
            break


for i in ctg:
    if not os.path.exists(result_path + i):
        os.makedirs(result_path + i)
    select(i)
    print(f"{i} selection is done!")

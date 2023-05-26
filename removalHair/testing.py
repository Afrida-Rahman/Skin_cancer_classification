import os
from glob import glob

import matplotlib
import numpy
from matplotlib import pyplot as plt
from PIL import Image
from hair_removal import remove_and_inpaint

os.chdir("..")


file_path = 'data/hair_rmd_train_raw_val_raw_test_70_20_10/train/'


def remove_hair(path):
    image = plt.imread(path)
    hairless_image, steps = remove_and_inpaint(image)
    os.remove(path)
    matplotlib.image.imsave(f"{path.split('.')[0]}.png", hairless_image)


ctg = ["akiec", "bcc", "bkl", "df", "mel", "vasc", "nv"]
c = 0
for i in ctg:
    files = glob(file_path + i + '/*')
    c = 0
    for j in files:
        remove_hair(j)
        c += 1
        print(f"Train : {i} class ->{c}")
    print(f"{i} class ->{j} is hair removed")

# ctg = ["akiec", "bcc", "bkl", "df", "mel", "vasc", "nv"]
# c = 0
# for i in ctg:
#     files = glob(file_path + i + '/*')
#     c = 0
# files = glob(file_path + '/*')
# sorted_images= sorted([j for j in files if j.split(".")[1]=="jpg"])
# print(len(sorted_images))
# sorted_images = numpy.load("sorted_list.npy")
# print(len(sorted_images))
# print("start -> 3000:5548")
# for i in sorted_images[3000:5548]:
#     remove_hair(i)
#     c += 1
#     print(f"{i} is hair removed -> {c}")

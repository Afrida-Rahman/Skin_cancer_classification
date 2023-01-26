# from transformers import ViTFeatureExtractor, TFViTForImageClassification, ViTForImageClassification
# import tensorflow as tf
# from datasets import load_dataset
# from hugsvision.dataio.VisionDataset import VisionDataset
# import numpy as np
#
# train, test, id2label, label2id = VisionDataset.fromImageFolder(
#     "raw_data/data_85_15_split/val/",
#     test_ratio=0.2,
#     balanced=False,
#     augmentation=False,
# )
# print(test)
#
# feature_extractor = ViTFeatureExtractor.from_pretrained("model_85_15_split/TRAIN_WITH_AUG/50_2022-11-23-17-41-32/feature_extractor/")
# model_85_15_split = ViTForImageClassification.from_pretrained("model_85_15_split/TRAIN_WITH_AUG/50_2022-11-23-17-41-32/model_85_15_split/")
#
# inputs = feature_extractor(test, return_tensors="pt")
# logits = model_85_15_split(**inputs).logits

# # model_85_15_split predicts one of the 1000 ImageNet classes
# predicted_label = int(tf.math.argmax(logits, axis=-1))
# print(model_85_15_split.config.id2label[predicted_label])

import os.path
from glob import glob
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, f1_score, recall_score, \
    precision_score
import numpy as np
import seaborn as sns
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
import matplotlib.pyplot as plt

path = "../HuggingFace/model/model_85_15_split/TRAIN_WITHOUT_AUG/50_2022-11-25-18-59-19/"

classifier = VisionClassifierInference(
    feature_extractor=ViTFeatureExtractor.from_pretrained(
        "feature_extractor"),
    model=ViTForImageClassification.from_pretrained("model_85_15_split"),
)


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


def separate_class_label(file_path, ctg):
    folders = glob(file_path + ctg + '/*')
    y_true, y_pred = [], []
    for i in folders:
        label = classifier.predict(img_path=i)
        y_true.append(i.split('/')[3])
        y_pred.append(label)
        # print("Predicted class:", label)
    return y_true, y_pred


ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
path = "../data/raw_data/data_85_15_split/val/"
a, b = [], []
for i in ctg:
    y_true, y_pred = separate_class_label(path, i)
    a.extend(y_true)
    b.extend(y_pred)
y_true = convert_label_to_int(a)
y_pred = convert_label_to_int(b)

np.save("../random_trial/y_true.npy", y_true)
np.save("../random_trial/y_pred.npy", y_pred)

# cf_matrix = confusion_matrix(y_true,y_pred)
# ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues') #/np.sum(cf_matrix), , fmt='.2%'
#
# ax.set_title('Confusion Matrix with labels\n\n')
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ')
#
# ax.xaxis.set_ticklabels(['akiec','bcc','bkl','df','mel','vasc','nv'])
# ax.yaxis.set_ticklabels(['akiec','bcc','bkl','df','mel','vasc','nv'])
#
# plt.show()

print(precision_score(y_true, y_pred, average="macro"))
print(recall_score(y_true, y_pred, average="macro"))
print(f1_score(y_true, y_pred, average="macro"))
print(accuracy_score(y_true, y_pred))

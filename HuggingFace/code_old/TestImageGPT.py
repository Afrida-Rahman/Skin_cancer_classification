import os

from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from transformers import ImageGPTForImageClassification, ImageGPTImageProcessor
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, \
    accuracy_score

model_folder = '/home/afrida/Documents/pProjects/Skin_cancer_classification/HuggingFace/model_85_15_split/IMAGEGPT_224_10E/10_2023-01-10-12-44-27/'
m_path = model_folder + "model_85_15_split/"
f_path = model_folder + "feature_extractor/"

result_path = "../result/result_85_15_split/"
test_data_path = "../../data/raw_data/data_70_20_10_split/test/"

epoch = 10
model_name = 'ImageGPT_Small'

classifier = VisionClassifierInference(
    feature_extractor=ImageGPTImageProcessor.from_pretrained(f_path),
    model=ImageGPTForImageClassification.from_pretrained(m_path),
)

ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def separate_class_label(file_path, ctg):
    folders = glob(file_path + ctg + '/*')
    y_true, y_pred = [], []
    for i in folders:
        print(i.split('/')[5])
        label = classifier.predict(img_path=i)
        y_true.append(i.split('/')[5])
        y_pred.append(label)
        print("Predicted class:", label)
    return y_true, y_pred


c, d = [], []
for i in ctg:
    a, b = separate_class_label(test_data_path, i)
    c.extend(a)
    d.extend(b)


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
        if i == "nv":
            a.append(5)
        if i == "vasc":
            a.append(6)
    return np.array(a)


y_true = convert_label_to_int(c)
y_pred = convert_label_to_int(d)

print(y_true.shape)
print(y_pred.shape)

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred,average="macro")
recall = recall_score(y_true, y_pred,average="macro")

classification_r = pd.DataFrame(classification_report(y_true, y_pred, target_names=ctg, output_dict=True))

df_cm = pd.DataFrame(cm, columns=ctg, index=ctg)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(result_path + "conf.jpg")

print(classification_r)

m = [pre,acc,recall]
df = pd.DataFrame(m, index=['pre','acc','recall'])

file_name = f"test_conf_{model_name}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1',result_path+"conf.jpg")
os.remove(result_path+"conf.jpg")
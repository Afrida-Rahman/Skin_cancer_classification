import json
import os

from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from transformers import AutoFeatureExtractor, ViTForImageClassification, ViTFeatureExtractor
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, \
    accuracy_score

model_folder = '/home/afrida/Documents/pProjects/Skin_cancer_classification/HuggingFace/model/VIT-CUSTOM_P9_R72_E10/10_2023-01-10-19-36-43/'
m_path = model_folder + "model/"
f_path = model_folder + "feature_extractor/"
t_path = model_folder + "trainer/"
result_path = "../result/"
test_data_path = "../../raw_data/train_test_valid_splitted/test/"
config_path = t_path + 'config.json'
cfg_file = open(config_path)
config = json.load(cfg_file)
epoch = 10
model_name = 'ViT-custom'
patch = config['patch_size']
resolution = config['image_size']

classifier = VisionClassifierInference(
    feature_extractor=ViTFeatureExtractor.from_pretrained(f_path),
    model=ViTForImageClassification.from_pretrained(m_path),
)

ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def separate_class_label(file_path, ctg):
    folders = glob(file_path + ctg + '/*')
    y_true, y_pred = [], []
    for i in folders:
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

# np.save(result_path+"y_true.npy", y_true)
# np.save(result_path+"y_pred.npy", y_pred)

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

cm = [pre,acc,recall]
df = pd.DataFrame(cm, index=['pre','acc','recall'])

file_name = f"test_conf_{model_name}_p{patch}_r{resolution}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1',result_path+"conf.jpg")
os.remove(result_path+"conf.jpg")

# akiec_label = np.zeros(np.shape(akiec_mat)[0])
# bcc_label = np.ones(np.shape(bcc_mat)[0])
# bkl_label = 2*np.ones(np.shape(bkl_mat)[0])
# df_label = 3*np.ones(np.shape(df_mat)[0])
# mel_label = 4*np.ones(np.shape(mel_mat)[0])
# nv_label = 5*np.ones(np.shape(nv_mat)[0])
# vasc_label = 6*np.ones(np.shape(vasc_mat)[0])
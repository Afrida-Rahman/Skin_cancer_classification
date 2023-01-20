import json
import os

import torch
from hugsvision.dataio import VisionDataset
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification, ConvNextConfig, ConvNextModel
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, \
    accuracy_score
from tqdm import tqdm
from hugsvision.dataio.VisionDataset import VisionDataset


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("..//..")
model_folder = 'HuggingFace/model/convNext/CONVNEXT_XL_AUG_P4_R224_E10/10_2023-01-13-12-09-08/'
m_path = model_folder + "model/"
f_path = model_folder + "feature_extractor/"
result_path = "HuggingFace/result/convNext/"
test_data_path = "aug_data/balanced/train_test_val/val/"

config_path = m_path + 'config.json'
# cfg_file = open(config_path)
# config = json.load(cfg_file)
config = ConvNextConfig.from_json_file(config_path)
print(config)
epoch = 10
model_name = 'ConvNext_XL_aug_b'
patch = config.patch_size
resolution = config.image_size

test, _, id2label, label2id = VisionDataset.fromImageFolder(
    test_data_path,
    test_ratio=0,
    balanced=False,
    augmentation=False,
)


feature_extractor=ConvNextFeatureExtractor.from_pretrained(f_path)
model=ConvNextForImageClassification.from_pretrained(m_path, config=config)
resolution=resolution

ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def evaluate(dataset):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_target = []

    # For each image
    for image, label in tqdm(dataset):
        # Compute
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Get predictions from the softmax layer
        preds = outputs.logits.softmax(1).argmax(1).tolist()
        all_preds.extend(preds)

        # Get hypothesis
        all_target.append(label)

    return all_target, all_preds

y_true, y_pred = evaluate(test)
# np.save(result_path+"y_true.npy", y_true)
# np.save(result_path+"y_pred.npy", y_pred)

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")

classification_r = pd.DataFrame(classification_report(y_true, y_pred, target_names=ctg, output_dict=True))

df_cm = pd.DataFrame(cm, columns=ctg, index=ctg)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(result_path + "conf.jpg")

print(classification_r)

cm = [pre, acc, recall]
df = pd.DataFrame(cm, index=['pre', 'acc', 'recall'])

file_name = f"Trail_val_conf_{model_name}_p{patch}_r{resolution}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1', result_path + "conf.jpg")
os.remove(result_path + "conf.jpg")

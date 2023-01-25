import os

import torch
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification, ConvNextConfig
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, \
    accuracy_score, roc_auc_score
from tqdm import tqdm
from hugsvision.dataio.VisionDataset import VisionDataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("..//..")
model_folder = 'HuggingFace/model/aug/convnext/CONVNEXT_L_AUG_B_P4_R224_E3/3_2023-01-16-13-00-55/'
m_path = model_folder + "model/"
f_path = model_folder + "feature_extractor/"
result_path = "HuggingFace/result/aug/convnext/"
test_data_path = "aug_data/balanced_MVT/train_test_val/val/"

config_path = m_path + 'config.json'
# cfg_file = open(config_path)
# config = json.load(cfg_file)
config = ConvNextConfig.from_json_file(config_path)
print(config)
epoch = 3
model_name = 'ConvNext_L_aug_b'
patch = config.patch_size
resolution = config.image_size

test, _, id2label, label2id = VisionDataset.fromImageFolder(
    test_data_path,
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

feature_extractor = ConvNextFeatureExtractor.from_pretrained(f_path)
model = ConvNextForImageClassification.from_pretrained(m_path, config=config)
resolution = resolution

ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


def evaluate(dataset):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_preds = []
    all_target = []
    all_pred_proba = []
    # For each image
    for image, label in tqdm(dataset):
        all_target.append(label)

        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        preds = outputs.logits.softmax(1).argmax(1).tolist()
        all_preds.extend(preds)

        softmax = torch.nn.Softmax(dim=0)
        pred_soft = softmax(outputs[0][0])
        probabilities = pred_soft.tolist()
        all_pred_proba.append(probabilities)

    return all_target, all_preds, all_pred_proba


y_true, y_pred, y_pred_proba = evaluate(test)

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
roc_auc = roc_auc_score(y_true, y_pred_proba, average="macro", multi_class='ovr')
classification_r = pd.DataFrame(classification_report(y_true, y_pred, target_names=ctg, output_dict=True))

df_cm = pd.DataFrame(cm, columns=ctg, index=ctg)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(result_path + "conf.jpg")

print(classification_r)

cm = [pre, acc, recall, roc_auc]
df = pd.DataFrame(cm, index=['pre', 'acc', 'recall', 'roc_auc'])

file_name = f"val_conf_{model_name}_p{patch}_r{resolution}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1', result_path + "conf.jpg")
os.remove(result_path + "conf.jpg")

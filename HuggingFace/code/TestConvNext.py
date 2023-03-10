import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
from PIL import Image
from hugsvision.dataio.VisionDataset import VisionDataset
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score, \
    roc_auc_score
from tqdm import tqdm
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification, ConvNextConfig

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("..//..")
d_type = "70_20_10"  # "85_15"
model_folder = 'HuggingFace/model/model_70_20_10_split/aug/convnext/XL_EACC_384R_3E_8B/3_2023-03-05-06-43-11/'
m_path = model_folder + "model/"
f_path = model_folder + "feature_extractor/"
# result_path = f"HuggingFace/result/result_{d_type}_split/aug/convnext/"
result_path = "HuggingFace/result/ISIC_2018/aug/convnext/"
# test_data_path = f"data/aug_data/data_{d_type}_split/384/test/"
test_data_path = "data/raw_data/ISIC_2018/384"
ext = 'test'
config_path = m_path + 'config.json'
# cfg_file = open(config_path)
# config = json.load(cfg_file)
config = ConvNextConfig.from_json_file(config_path)
epoch = 3
model_name = 'XL_eacc'
patch = config.patch_size
resolution = config.image_size

# test, _, _, _ = VisionDataset.fromImageFolder(
#     test_data_path,
#     test_ratio=0,
#     balanced=False,
#     augmentation=False,
# )

test = glob(test_data_path + '/*')

feature_extractor = ConvNextFeatureExtractor(do_normalize=True).from_pretrained(f_path)
model = ConvNextForImageClassification.from_pretrained(m_path)  # , config=config
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

def evaluate_isic(dataset):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_preds = []
    label_file = pd.read_csv("data/raw_data/ISIC_2018/labels.csv")
    image_str = label_file['image_id']
    image_label = label_file['dx']
    ctg = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
    image_label_dict = {}
    for i in range(len(image_str)):
        image_label_dict[image_str[i] + ".jpg"] = ctg[image_label[i]]

    print(image_label_dict)
    all_target = []
    all_pred_proba = []
    # For each image
    for file in tqdm(dataset):
        file_str = file.split('/')[4]
        all_target.append(image_label_dict[file_str])
        image = Image.open(file)
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        preds = outputs.logits.softmax(1).argmax(1).tolist()
        all_preds.extend(preds)

        softmax = torch.nn.Softmax(dim=0)
        pred_soft = softmax(outputs[0][0])
        probabilities = pred_soft.tolist()
        all_pred_proba.append(probabilities)

    return all_target, all_preds, all_pred_proba

# y_true, y_pred, y_pred_proba = evaluate(test)
y_true, y_pred, y_pred_proba = evaluate_isic(test)

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, average="macro")
recall = recall_score(y_true, y_pred, average="macro")
roc_auc = roc_auc_score(y_true, y_pred_proba, average="macro", multi_class='ovr')
# fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba, pos_label=2)
# pr_auc = auc(fpr, tpr)
# Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# n_classes = 7
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_true[:,i],y_pred_proba[:,i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Plot of a ROC curve for a specific class
# for i in range(n_classes):
#     plt.figure()
#     plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.savefig(result_path + "roc_curve.jpg")
classification_r = pd.DataFrame(classification_report(y_true, y_pred, target_names=ctg, output_dict=True))

df_cm = pd.DataFrame(cm, columns=ctg, index=ctg)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(result_path + "conf.jpg")

print(classification_r)

cm = [pre, acc, recall, roc_auc]
df = pd.DataFrame(cm, index=['pre', 'acc', 'recall', 'roc_auc'])

file_name = f"{ext}_conf_{model_name}_p{patch}_r{resolution}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1', result_path + "conf.jpg")
os.remove(result_path + "conf.jpg")

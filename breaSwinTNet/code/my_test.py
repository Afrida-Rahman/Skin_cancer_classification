#https://machinelearningmastery.com/check-point-deep-learning-models-keras/
import os

import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from openpyxl import Workbook
wb = Workbook()
from openpyxl.drawing.image import Image
s2 = wb.create_sheet("confusion_matrics", 0) # insert at first position

print("GPU: " + str(tf.config.list_physical_devices('GPU')))
print("Available gpu: " + str(tf.test.is_gpu_available()))

data_path = "/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/customized_data/"
result_path = "/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/result/"
model_path = "/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/model/"

lr = .01
epoch = 20
sd = 201
drp = .05
# name =
model = tf.keras.models.load_model(model_path + f"small_{epoch}e_{sd}sd_{lr}lr_{drp}drp")
x_test = np.load(data_path + "x_test.npy")
y_test = np.load(data_path + "y_test.npy")

model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
y_prob = np.argmax(y_pred, axis=1)

acc = sklearn.metrics.accuracy_score(y_test, y_prob)
auc = sklearn.metrics.roc_auc_score(y_test, y_pred, multi_class='ovo')
target_names = ["akiec", "bcc", "bkl", "df", "mel", "vasc", "nv"]
f1 = sklearn.metrics.f1_score(y_test, y_prob, average='micro')
bacc = sklearn.metrics.balanced_accuracy_score(y_test, y_prob)
mcc = sklearn.metrics.matthews_corrcoef(y_test, y_prob)
cf = pd.DataFrame(sklearn.metrics.classification_report(y_test, y_prob, target_names=target_names, output_dict=True))
cf_matrix = confusion_matrix(y_test, y_prob)

print(f'Accuracy {acc}')
print(f'Balanced accuracy: {bacc}')
print(f'AUC: {auc}')
print(f'F1-score : {f1}')
print(f'MCC: {mcc}')
print(f'{cf}')

metrics = [acc, bacc, auc, f1, mcc]
df = pd.DataFrame(metrics, index=['acc','bacc','auc','f1','mcc'])


labels = ['akiec','bcc','bkl','df','mel','vasc','nv']
df_cm = pd.DataFrame(cf_matrix, index = labels, columns = labels)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="", cmap="Blues")
plt.savefig(result_path+"conf.jpg")

file_name = "output_"+str(epoch)+"e_"+str(sd)+"sd_"+str(lr)+"lr_"+str(drp)+"drp.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    cf.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1',result_path+"conf.jpg")

os.remove(result_path+"conf.jpg")
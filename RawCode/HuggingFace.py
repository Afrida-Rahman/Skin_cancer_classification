# source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353

from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np


train, test, id2label, label2id = VisionDataset.fromImageFolder(
    "raw_data/data_85_15_split/train/",
    test_ratio=0.2,
    balanced=False,
    augmentation=False,
)

huggingface_model = 'google/vit-base-patch16-224-in21k'

trainer = VisionClassifierTrainer(
    model_name="Train_With_Aug",
    train=train,
    test=test,
    output_dir="../model_85_15_split/",
    max_epochs=50,
    batch_size=32,  # On RTX 2080 Ti
    lr=2e-5,
    fp16=True,
    model=ViTForImageClassification.from_pretrained(
        huggingface_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    ),
    feature_extractor=ViTFeatureExtractor.from_pretrained(
        huggingface_model,
    ),
)

ref, hyp = trainer.evaluate_f1_score()
print("ref : ")
print(ref)
print("\nHyp: ")
print(hyp)

#
# test_img = np.load("raw_data/test_img_72x72.npy")
# test_label = np.load("raw_data/test_label_72x72.npy")
# print("start testing")
# trainer.testing("raw_data/data_85_15_split/val/nv/ISIC_0024309.jpg", "0")
# trainer.test("raw_data/data_85_15_split/val/")

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ref, hyp)
labels = list(label2id.keys())
df_cm = pd.DataFrame(cm, index = labels, columns = labels)

plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig("./imgs/conf_matrix_1.jpg")
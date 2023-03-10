# # source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
# # Xray source: https://huggingface.co/blog/vision-transformers
# # https://github.com/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb
# 
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import AutoImageProcessor, ViTMSNForImageClassification
import numpy as np
from transformers import ViTMSNConfig, ViTMSNModel

train, _, id2label, label2id = VisionDataset.fromImageFolder(
    "../../raw_data/data_70_20_10_split/train/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

test, _, _, _ = VisionDataset.fromImageFolder(
    "../../raw_data/data_70_20_10_split/val/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

# Initializing a ViT vit-base-patch16-224 style configuration
# configuration = ViTConfig(hidden_size=1024, num_hidden_layers=24, intermediate_size=4096, num_attention_heads=16,
#                           patch_size=32, image_size=384)
configuration = ViTMSNConfig()

# Initializing a model_85_15_split (with random weights) from the vit-base-patch16-224 style configuration
model = ViTMSNModel(configuration)
huggingface_model= 'facebook/vit-msn-large'
epoch = 10
model_name = 'ViT-MSN'
resolution = 'large'
trainer = VisionClassifierTrainer(
    model_name=f"{model_name}_{resolution}_e{epoch}",
    train=train,
    test=test,
    output_dir="../model/model_85_15_split/",
    max_epochs=epoch,
    batch_size=8,  # On RTX 2080 Ti
    lr=1e-5,
    fp16=False,
    model=ViTMSNForImageClassification.from_pretrained
    (
        huggingface_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    ),
    feature_extractor= AutoImageProcessor.from_pretrained(huggingface_model)
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
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(f"../result/val_conf_{model_name}_{resolution}_e{epoch}.jpg")

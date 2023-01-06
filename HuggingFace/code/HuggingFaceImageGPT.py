# # source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
# # Xray source: https://huggingface.co/blog/vision-transformers
# # https://github.com/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb
# 
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ImageGPTImageProcessor, ImageGPTForImageClassification
import numpy as np
from transformers import ImageGPTConfig, ImageGPTModel

train, _, id2label, label2id = VisionDataset.fromImageFolder(
    "../../raw_data/train_test_valid_splitted/train/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

test, _, _, _ = VisionDataset.fromImageFolder(
    "../../raw_data/train_test_valid_splitted/val/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

# Initializing a ViT vit-base-patch16-224 style configuration
configuration = ImageGPTConfig()

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
model = ImageGPTModel(configuration)
epoch = 10
huggingface_model = 'openai/imagegpt-small'
trainer = VisionClassifierTrainer(
    model_name="ImageGPT_224_Train_Without_Aug"+str(epoch)+"e",
    train=train,
    test=test,
    output_dir="../model/",
    max_epochs=epoch,
    batch_size=8,  # On RTX 2080 Ti
    lr=2e-5,
    fp16=False,
    model=ImageGPTForImageClassification.from_pretrained(huggingface_model,
                                                    num_labels=len(label2id),
                                                    label2id=label2id,
                                                    id2label=id2label
                                                    ),
    feature_extractor= ImageGPTImageProcessor.from_pretrained(huggingface_model)
)

ref, hyp = trainer.evaluate_f1_score()

#
# test_img = np.load("raw_data/test_img_72x72.npy")
# test_label = np.load("raw_data/test_label_72x72.npy")
# print("start testing")
# trainer.testing("raw_data/train_test_splitted/val/nv/ISIC_0024309.jpg", "0")
# trainer.test("raw_data/train_test_splitted/val/")

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ref, hyp)
labels = list(label2id.keys())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="", cmap="Blues")
plt.savefig("../result/training_conf_"+str(epoch)+"e_no_aug_image-gpt_HAM10k_validation_224.jpg")

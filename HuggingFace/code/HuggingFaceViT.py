# # source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
# # Xray source: https://huggingface.co/blog/vision-transformers
# # https://github.com/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb
#
import os

from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoFeatureExtractor
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import PIL
from transformers import ViTConfig, ViTModel

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

epoch = 10
model_name = 'ViT-custom'

if model_name.__contains__("custom"):
    # Initializing a ViT vit-base-patch16-224 style configuration
    configuration = ViTConfig(hidden_size=1024, num_hidden_layers=24, intermediate_size=4096, num_attention_heads=16,
                              patch_size=9, image_size=72, id2label=id2label, label2id=label2id,
                              num_labels=len(label2id))

    # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
    model = ViTModel(configuration)
    configuration = model.config
    patch = model.config.patch_size
    resolution = model.config.image_size

    trainer = VisionClassifierTrainer(
        model_name=f"{model_name}_p{patch}_r{resolution}_e{epoch}",
        train=train,
        test=test,
        output_dir="../model/",
        max_epochs=epoch,
        batch_size=8,  # On RTX 2080 Ti
        lr=1e-5,
        fp16=False,
        model=ViTForImageClassification(config=configuration),
        feature_extractor= ViTFeatureExtractor(do_resize=True, size=72, do_normalize=True, resample=PIL.Image.Resampling.NEAREST)
    )
else:
    pretrained_model = 'google/vit-large-patch32-384'
    patch = 32
    resolution = 384
    trainer = VisionClassifierTrainer(
        model_name=f"{model_name}_p{patch}_r{resolution}_e{epoch}",
        train=train,
        test=test,
        output_dir="../model/",
        max_epochs=epoch,
        batch_size=4,  # On RTX 2080 Ti
        lr=2e-5,
        fp16=False,
        model=ViTForImageClassification.from_pretrained(pretrained_model,
                                                        num_labels=len(label2id),
                                                        label2id=label2id,
                                                        id2label=id2label,
                                                        ignore_mismatched_sizes=True
                                                        ),
        feature_extractor=ViTFeatureExtractor.from_pretrained(pretrained_model)
    )
trainer.training_args.load_best_model_at_end = True
ref, hyp = trainer.evaluate_f1_score()

result_path = "../result/"
ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

cm = confusion_matrix(y_true=ref, y_pred=hyp)
acc = accuracy_score(y_true=ref, y_pred=hyp)
pre = precision_score(y_true=ref, y_pred=hyp, average="macro")
recall = recall_score(y_true=ref, y_pred=hyp, average="macro")

classification_r = pd.DataFrame(classification_report(ref, hyp, target_names=ctg, output_dict=True))

df_cm = pd.DataFrame(cm, columns=ctg, index=ctg)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(result_path + "conf.jpg")

print(classification_r)

m = [pre,acc,recall]
df = pd.DataFrame(m, index=['pre','acc','recall'])

file_name = f"val_conf_{model_name}_p{patch}_r{resolution}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1',result_path+"conf.jpg")
os.remove(result_path+"conf.jpg")

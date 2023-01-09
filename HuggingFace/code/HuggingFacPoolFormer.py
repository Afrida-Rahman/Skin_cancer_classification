# # source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
# # Xray source: https://huggingface.co/blog/vision-transformers
# # https://github.com/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb
#
import os
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import PoolFormerFeatureExtractor, PoolFormerForImageClassification
from transformers import PoolFormerConfig, PoolFormerModel
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

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
# configuration = ViTConfig(hidden_size=1024, num_hidden_layers=24, intermediate_size=4096, num_attention_heads=16,
#                           patch_size=32, image_size=384)
# configuration = PoolFormerConfig(
#     num_channels = 3,
#     patch_size = 16,
#     stride = 16,
#     pool_size = 3,
#     mlp_ratio = 4.0,
#     depths = [2, 2, 6, 2],
#     hidden_sizes = [64, 128, 320, 512],
#     patch_sizes = [7, 3, 3, 3],
#     strides = [4, 2, 2, 2],
#     padding = [2, 1, 1, 1],
#     num_encoder_blocks = 4,
#     drop_path_rate = 0.0,
#     hidden_act = 'gelu',
#     use_layer_scale = True,
#     layer_scale_init_value = 1e-05,
#     initializer_range = 0.02
# )

configuration = PoolFormerConfig()

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
model = PoolFormerModel(configuration)
configuration = model.config
huggingface_model= 'sail/poolformer_m48'
epoch = 5
model_name = 'PoolFormer'
model_variant = 'm48'
trainer = VisionClassifierTrainer(
    model_name=f"{model_name}_{model_variant}_e{epoch}",
    train=train,
    test=test,
    output_dir="../model/",
    max_epochs=epoch,
    batch_size=8,  # On RTX 2080 Ti
    lr=1e-5,
    fp16=False,
    model=PoolFormerForImageClassification.from_pretrained
    (
        huggingface_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    ),
    feature_extractor= PoolFormerFeatureExtractor.from_pretrained(huggingface_model)
)

ref, hyp = trainer.evaluate_f1_score()
print("ref : ")
print(ref)
print("\nHyp: ")
print(hyp)


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

file_name = f"val_conf_{model_name}_{model_variant}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1',result_path+"conf.jpg")
os.remove(result_path+"conf.jpg")

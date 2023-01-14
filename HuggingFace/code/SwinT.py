import os

import PIL
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from transformers import SwinConfig, Swinv2Config, SwinModel, Swinv2Model
from transformers import SwinForImageClassification, Swinv2ForImageClassification, AutoFeatureExtractor

os.chdir("..//..")
# train, _, id2label, label2id = VisionDataset.fromImageFolder(
#     "../../raw_data/train_test_valid_splitted/train/",
#     test_ratio=0,
#     balanced=False,
#     augmentation=False,
# )
#
# test, _, _, _ = VisionDataset.fromImageFolder(
#     "../../raw_data/train_test_valid_splitted/val/",
#     test_ratio=0,
#     balanced=False,
#     augmentation=False,
# )

train, _, id2label, label2id = VisionDataset.fromImageFolder(
    "aug_data/balanced/train_test_val/train/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

test, _, _, _ = VisionDataset.fromImageFolder(
    "aug_data/balanced/train_test_val/val/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

epoch = 5
model_name = 'Swin_L_aug_balanced'

# for pretrained model only
model_path = "HuggingFace/model/swin/"
result_path = "HuggingFace/result/swin/"
pretrained_model = 'microsoft/swin-large-patch4-window12-384-in22k'
patch = 4
resolution = 384
window = 12

if model_name.__contains__("custom"):
    result_path = "HuggingFace/result/swin/"
    model_path = "HuggingFace/model/swin/"
    configuration = Swinv2Config(
            image_size=192,
            patch_size=16,
            num_channels=3,
            embed_dim=64,
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 8, 16],
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.1,
            hidden_act='gelu',
            use_absolute_embeddings=False,
            patch_norm=True,
            initializer_range=0.02,
            layer_norm_eps=1e-05,
            encoder_stride=32
        )

    model = SwinModel(configuration)

    configuration = model.config
    patch = model.config.patch_size
    resolution = model.config.image_size

    trainer = VisionClassifierTrainer(
        model_name=f"{model_name}_p{patch}_w{window}_r{resolution}_e{epoch}",
        train=train,
        test=test,
        output_dir=model_path,
        max_epochs=epoch,
        batch_size=8,  # On RTX 2080 Ti
        lr=1e-4,
        fp16=False,
        model=SwinForImageClassification(config=configuration),
        feature_extractor=AutoFeatureExtractor(do_resize=True, size=72, do_normalize=True,
                                              resample=PIL.Image.Resampling.NEAREST)
    )

else:

    trainer = VisionClassifierTrainer(
        model_name=f"{model_name}_p{patch}_w{window}_r{resolution}_e{epoch}",
        train=train,
        test=test,
        output_dir=model_path,
        max_epochs=epoch,
        batch_size=1,  # On RTX 2080 Ti
        lr=1e-4,
        fp16=True,
        model=SwinForImageClassification.from_pretrained(pretrained_model,
                                                        num_labels=len(label2id),
                                                        label2id=label2id,
                                                        id2label=id2label,
                                                        ignore_mismatched_sizes=True
                                                        ),
        feature_extractor=AutoFeatureExtractor.from_pretrained(pretrained_model)
    )
# trainer.training_args.load_best_model_at_end = True
ref, hyp = trainer.evaluate_f1_score()

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

m = [pre, acc, recall]
df = pd.DataFrame(m, index=['pre', 'acc', 'recall'])

file_name = f"val_conf_{model_name}_p{patch}_w{window}_r{resolution}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1', result_path + "conf.jpg")
os.remove(result_path + "conf.jpg")

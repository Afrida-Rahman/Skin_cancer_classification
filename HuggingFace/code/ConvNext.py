# # source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
# # Xray source: https://huggingface.co/blog/vision-transformers
# # https://github.com/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb
# Accuracy= https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
# model link: https://huggingface.co/models?sort=downloads&search=convN
import os

from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ConvNextForImageClassification, ConvNextFeatureExtractor
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
import PIL
from transformers import ConvNextConfig, ConvNextModel
# from HuggingFace.code.VisionTrainerCustom import VisionTrainerCustom

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("..//..")

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

epoch = 20
model_name = 'ConvNext_L_aug_b'

# for pretrained model only
model_path = "HuggingFace/model/convNext/"
result_path = "HuggingFace/result/convNext/"
pretrained_model = 'facebook/convnext-large-224-22k-1k'
patch = 4
resolution = 224

if model_name.__contains__("custom"):
    print('CUSTOM RUNNING .....')

    configuration = ConvNextConfig()
    model = ConvNextModel(configuration)

    configuration = model.config
    patch = model.config.patch_size
    resolution = model.config.image_size

    trainer = VisionClassifierTrainer(
        model_name=f"{model_name}_p{patch}_r{resolution}_e{epoch}",
        train=train,
        test=test,
        output_dir=model_path,
        max_epochs=epoch,
        batch_size=8,  # On RTX 2080 Ti
        lr=1e-4,
        fp16=True,
        eval_metric="eval_loss",
        model=ConvNextForImageClassification(config=configuration),
        feature_extractor=ConvNextFeatureExtractor(do_resize=True, size=72, do_normalize=True,
                                                   resample=PIL.Image.Resampling.NEAREST)
    )
else:
    print("pretrained running ....")
    trainer = VisionClassifierTrainer(
        model_name=f"{model_name}_p{patch}_r{resolution}_e{epoch}",
        train=train,
        test=test,
        output_dir=model_path,
        max_epochs=epoch,
        batch_size=4,  # On RTX 2080 Ti
        lr=1e-4,
        fp16=False,
        model=ConvNextForImageClassification.from_pretrained(pretrained_model,
                                                             num_labels=len(label2id),
                                                             label2id=label2id,
                                                             id2label=id2label,
                                                             ignore_mismatched_sizes=True
                                                             ),
        feature_extractor=ConvNextFeatureExtractor.from_pretrained(pretrained_model),
        classification_report_digits=4,
        eval_metric="eval_loss"

    )
ref, hyp = trainer.evaluate_f1_score()
# ref, hyp = trainer.evaluate()

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

file_name = f"val_conf_{model_name}_p{patch}_r{resolution}_e{epoch}.xlsx"
with pd.ExcelWriter(result_path + file_name) as writer:
    df.to_excel(writer, sheet_name='all_metrics')
    classification_r.to_excel(writer, sheet_name='metrics_with_labels')
    worksheet = writer.sheets['all_metrics']
    worksheet.insert_image('E1', result_path + "conf.jpg")
os.remove(result_path + "conf.jpg")

## Sanity Check

print("Start testing .....")
print()
ind_test, _, _, _ = VisionDataset.fromImageFolder(
    "aug_data/balanced/train_test_val/test/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)
test_ref, test_hyp = trainer.evaluate(dataset=ind_test)
classification_r = pd.DataFrame(classification_report(test_ref, test_hyp, target_names=ctg, output_dict=True))
print(classification_r)

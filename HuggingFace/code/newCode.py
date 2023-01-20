# # source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
# # Xray source: https://huggingface.co/blog/vision-transformers
# # https://github.com/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb
# Accuracy= https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
# model link: https://huggingface.co/models?sort=downloads&search=convN

import os
from transformers import ConvNextForImageClassification, ConvNextFeatureExtractor
from transformers import ConvNextConfig, ConvNextModel
from Training import Training

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("..//..")

epoch = 1
model_name = 'ConvNext_L_aug_b'
data_path = "aug_data/balanced/train_test_val"
model_path = "HuggingFace/model/convNext/"
result_path = "HuggingFace/result/convNext/"
pretrained_model = 'facebook/convnext-large-224-22k-1k'
patch = 4
resolution = 224
batch=2
model = ConvNextModel(ConvNextConfig())
configuration = model.config

train, _, id2label, label2id = Training().read_image(path=data_path + "/train/")
test, _, _, _ = Training().read_image(path=data_path + "/val/")
print("Train test obtained.")

model = ConvNextForImageClassification.from_pretrained(pretrained_model, num_labels=len(label2id), label2id=label2id,
                                                       id2label=id2label, ignore_mismatched_sizes=True)
feature_extractor = ConvNextFeatureExtractor.from_pretrained(pretrained_model)

training = Training(train=train, test=test, model_name=model_name, model_path=model_path, epoch=epoch, batch=batch,
                    model=model, feature_extractor=feature_extractor, eval_metric="eval_loss",id2label=id2label, label2id=label2id)
training.build_trainer()
print("trainer ready")
y_true, y_pred = training.evaluate_f1_score()
print("prediction ready")
training.compute_metrics(y_true=y_true, y_pred=y_pred, result_path=result_path, ext="val")

## Sanity Check

# print("Start testing .....")
# print()
# ind_test, _, _, _ = ReadImage("/test/")
# test_ref, test_hyp = trainer.evaluate(dataset=ind_test)
# classification_r = pd.DataFrame(classification_report(test_ref, test_hyp, target_names=ctg, output_dict=True))
# print(classification_r)
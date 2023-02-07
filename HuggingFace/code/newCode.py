# source: https://medium.com/@yanis.labrak/how-to-train-a-custom-vision-transformer-vit-image-classifier-to-help-endoscopists-in-under-5-min-2e7e4110a353
# Xray source: https://huggingface.co/blog/vision-transformers
# https://github.com/qanastek/HugsVision/blob/main/recipes/kvasir_v2/binary_classification/Kvasir_v2_Image_Classifier.ipynb
# Accuracy= https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
# model_85_15_split link: https://huggingface.co/models?sort=downloads&search=convN

import os

from transformers import ConvNextConfig, ConvNextModel
from transformers import ConvNextForImageClassification, ConvNextFeatureExtractor

from Training import Training

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("..//..")

epoch = 5
d_type = "70_20_10"  # "85_15"
train_data_path = f"data/aug_data/data_{d_type}_split/balanced/train/"

# test_data_path = f"data/raw_data/data_{d_type}_split/val/"
model_path = f"HuggingFace/model/model_{d_type}_split/aug/convnext/trial/balanced/"
result_path = f"HuggingFace/result/result_{d_type}_split/aug/convnext/trial/balanced/"

pretrained_model = 'facebook/convnext-large-224-22k-1k'
patch = 4
resolution = 224
batch = 8
model = ConvNextModel(ConvNextConfig())
configuration = model.config
model_name = f'ConvNext_L_eloss_{resolution}r_{epoch}e_{batch}b'

train, test, id2label, label2id = Training().read_image(path=train_data_path, test_ratio=0.2)
# test, _, _, _ = Training().read_image(path=test_data_path, test_ratio=0)
print("Train test obtained.")

model = ConvNextForImageClassification.from_pretrained(pretrained_model, num_labels=len(label2id), label2id=label2id,
                                                       id2label=id2label, ignore_mismatched_sizes=True)
feature_extractor = ConvNextFeatureExtractor(do_normalize=True).from_pretrained(pretrained_model)

training = Training(train=train, test=test, model_name=model_name, model_path=model_path, epoch=epoch, batch=batch,
                    model=model, feature_extractor=feature_extractor, eval_metric="eval_loss", is_best_model=True,
                    evaluation_strategy="steps", save_strategy="steps", id2label=id2label, label2id=label2id, fp16=True)
training.build_trainer()
print("trainer ready")
y_true, y_pred, y_pred_proba = training.evaluate_f1_score()
print("prediction ready")
training.compute_eval_metrics(y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba, result_path=result_path,
                              ext="val")

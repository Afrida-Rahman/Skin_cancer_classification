import os

from transformers import ConvNextForImageClassification, ConvNextImageProcessor

from Training import Training

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("..//..")

epoch = 5
train_data_path = f"data/fr_data/pytorch/data_70_20_10_split/384/"
# test_data_path = f"data/raw_data/data_{d_type}_split/val/"
model_path = f"HuggingFace/model/model_70_20_10_split/raw/convnext/"
result_path = f"HuggingFace/result/result_70_20_10_split/raw/convnext/"

pretrained_model = "facebook/convnext-base-384-22k-1k"  # 'facebook/convnext-large-224-22k-1k'  # "google/vit-large-patch16-224-in21k"
patch = 4
resolution = 384
batch = 4

# model = ConvNextModel(ConvNextConfig())
# configuration = model.config
model_name = f'convnext_b_recheck_{resolution}r_{epoch}e_{batch}b'

train, _, id2label, label2id = Training().read_image(path=train_data_path + 'train/', test_ratio=0)
test, _, _, _ = Training().read_image(path=train_data_path + "val/", test_ratio=0)
print("Train test obtained.")

model = ConvNextForImageClassification.from_pretrained(pretrained_model, num_labels=len(label2id), label2id=label2id,
                                                       id2label=id2label, ignore_mismatched_sizes=True)
feature_extractor = ConvNextImageProcessor().from_pretrained(
    pretrained_model)  # ConvNextFeatureExtractor(do_normalize=True, size=resolution, do_rescale=True)

training = Training(train=train, test=test, model_name=model_name, model_path=model_path, epoch=epoch, batch=batch,
                    model=model, feature_extractor=feature_extractor, eval_metric="accuracy", is_best_model=True,
                    evaluation_strategy="epoch", save_strategy="epoch", id2label=id2label, label2id=label2id, fp16=True,
                    lr=2e-5)
training.build_trainer()
print("trainer ready")
y_val_true, y_val_pred, y_val_pred_proba, y_train_true, y_train_pred, y_train_pred_proba = training.evaluate_f1_score()
print("prediction ready")
training.compute_eval_metrics(y_true=y_val_true, y_pred=y_val_pred, y_pred_proba=y_val_pred_proba,
                              result_path=result_path,
                              ext="val")
training.compute_eval_metrics(y_true=y_train_true, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba,
                              result_path=result_path,
                              ext="val")

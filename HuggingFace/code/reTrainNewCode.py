import os

from transformers import ConvNextForImageClassification, ConvNextImageProcessor

from Training import Training

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.chdir("..//..")

train_data_path = f"data/fr_data/pytorch/data_70_20_10_split/384/"
model_path = f"HuggingFace/model/model_70_20_10_split/fr/"
result_path = f"HuggingFace/result/result_fr_data/"

trained_model = "HuggingFace/model/model_70_20_10_split/fr/CONVNEXT_B_CHECK_EACC_384R_5E_4B/5_2023-03-20-18-07-38"
resolution = 384
epoch = 5
batch = 4

model_name = f'convnext_b_recheck_{resolution}r_{epoch}e_{batch}b'

train, _, id2label, label2id = Training().read_image(path=train_data_path + 'train/', test_ratio=0)
test, _, _, _ = Training().read_image(path=train_data_path + "val/", test_ratio=0)
print("Train test obtained.")

model = ConvNextForImageClassification.from_pretrained(trained_model+'/model', num_labels=len(label2id), label2id=label2id,
                                                       id2label=id2label, ignore_mismatched_sizes=True)
feature_extractor = ConvNextImageProcessor().from_pretrained(trained_model+'/feature_extractor')

training = Training(train=train, test=test, model_name=model_name, model_path=model_path, epoch=epoch, batch=batch,
                    model=model, feature_extractor=feature_extractor, eval_metric="accuracy", is_best_model=True,
                    evaluation_strategy="epoch", save_strategy="epoch", id2label=id2label, label2id=label2id, fp16=True,
                    lr=2e-5)
training.build_trainer()
print("trainer ready")

y_val_true, y_val_pred, y_val_pred_proba, y_train_true, y_train_pred, y_train_pred_proba = training.evaluate_f1_score()
print("prediction ready")

training.compute_eval_metrics(y_true=y_val_true, y_pred=y_val_pred, y_pred_proba=y_val_pred_proba,
                              result_path=result_path, ext="val")
training.compute_eval_metrics(y_true=y_train_true, y_pred=y_train_pred, y_pred_proba=y_train_pred_proba,
                              result_path=result_path, ext="train")

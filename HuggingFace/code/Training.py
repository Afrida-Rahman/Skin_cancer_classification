import os
from datetime import datetime

import numpy
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchmetrics
from hugsvision.dataio.ImageClassificationCollator import ImageClassificationCollator
from hugsvision.dataio.VisionDataset import VisionDataset
from imblearn.metrics import sensitivity_score, specificity_score
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, \
    roc_auc_score, f1_score, top_k_accuracy_score
from tqdm import tqdm
from transformers import Trainer
from transformers.training_args import TrainingArguments


# class CustomCallback(TrainerCallback):
#
#     def __init__(self, trainer) -> None:
#         super().__init__()
#         self._trainer = trainer
#
#     def on_epoch_end(self, args, state, control, **kwargs):
#         if control.should_evaluate:
#             control_copy = deepcopy(control)
#             self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
#             return control_copy
#

class Training:

    def __init__(self, train="", test="", id2label={}, label2id={}, model_name="", model_path="", epoch=0, batch=8,
                 model="",
                 feature_extractor="", config="", resolution=224, pretrained_model="", patch=0, lr=1e-4, fp16=True,
                 eval_metric="accuracy", is_best_model=True, balanced=False, augmentation=False,
                 save_total_limit=1, weight_decay=0.01, save_steps=10000, evaluation_strategy='epoch',
                 save_strategy='steps', classification_report_digits=4):

        self.train = train
        self.test = test
        self.id2label = id2label
        self.label2id = label2id
        self.model_name = model_name
        self.model_path = model_path
        self.epoch = epoch
        self.batch = batch
        self.model = model
        self.config = config
        self.feature_extractor = feature_extractor
        self.patch = patch
        self.pretrained_model = pretrained_model
        self.resolution = resolution
        self.lr = lr
        self.fp16 = fp16
        self.eval_metric = eval_metric
        self.is_best_model = is_best_model
        self.balanced = balanced
        self.augmentation = augmentation
        self.save_total_limit = save_total_limit
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.classification_report_digits = classification_report_digits

    def read_image(self, path, test_ratio, augmentation=False):
        data, _, id2label, label2id = VisionDataset.fromImageFolder(
            path,
            test_ratio=test_ratio,
            balanced=self.balanced,
            augmentation=augmentation,
        )
        return data, _, id2label, label2id

    def build_trainer(self, resume=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.metric = torchmetrics.Accuracy('multiclass', num_classes=len(list(self.label2id.keys())))
        self.collator = ImageClassificationCollator(self.feature_extractor)
        self.model_path = self.__getOutputPath()
        self.__openLogs()

        self.training_args = TrainingArguments(
            output_dir=self.model_path,
            save_total_limit=self.save_total_limit,
            weight_decay=self.weight_decay,
            save_steps=self.save_steps,
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch,
            per_device_eval_batch_size=self.batch,
            num_train_epochs=self.epoch,
            metric_for_best_model=self.eval_metric,
            logging_dir=self.model_path,
            evaluation_strategy=self.evaluation_strategy,
            load_best_model_at_end=self.is_best_model,
            overwrite_output_dir=True,
            fp16=self.fp16,
            save_strategy=self.save_strategy
        )

        print("Trainer building...")
        self.trainer = Trainer(
            self.model,
            self.training_args,
            train_dataset=self.train,
            eval_dataset=self.test,
            data_collator=self.collator,
            compute_metrics=self.compute_metrics
        )

        print("Start Training!")
        # self.trainer.add_callback(CustomCallback(self.trainer))
        self.trainer.train(resume_from_checkpoint=False)

        print("Start Saving Model....")
        self.trainer.save_model(self.model_path + "/trainer/")
        self.model.save_pretrained(self.model_path + "/model/")
        self.feature_extractor.save_pretrained(self.model_path + "/feature_extractor/")
        print("Model saved at: \033[93m" + self.model_path + "\033[0m")

        self.logs_file.close()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        res_dict = {'accuracy': metrics.accuracy_score(y_true=labels, y_pred=predictions)}
        return res_dict

    # def compute_metrics(eval_pred):
    #     top_k = 3
    #     predictions, labels = eval_pred
    #     preds = np.argsort(-predictions)[:, 0:top_k]
    #     acc_at_k = sum([l in p for l, p in zip(labels, preds)]) / len(labels)
    #     return {'acc_at_k': acc_at_k}

    def compute_eval_metrics(self, y_true, y_pred, y_pred_proba, result_path, ext):
        ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        numpy.save(result_path + "y_pred.npy", y_pred)
        numpy.save(result_path + "y_true.npy", y_true)
        numpy.save(result_path + "y_pred_proba.npy", y_pred_proba)

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        pre = precision_score(y_true=y_true, y_pred=y_pred, average="macro")
        recall = recall_score(y_true=y_true, y_pred=y_pred, average="macro")
        classification_r = pd.DataFrame(classification_report(y_true, y_pred, target_names=ctg, output_dict=True))
        roc_auc = roc_auc_score(y_true, y_pred_proba, average="macro", multi_class='ovr')
        spe = specificity_score(y_true=y_true, y_pred=y_pred, average="macro")
        sns = sensitivity_score(y_true=y_true, y_pred=y_pred, average="macro")
        f1_s = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        top_1_acc = top_k_accuracy_score(y_true=y_true, y_score=y_pred_proba, k=1, normalize=True)
        top_2_acc = top_k_accuracy_score(y_true=y_true, y_score=y_pred_proba, k=2, normalize=True)
        top_3_acc = top_k_accuracy_score(y_true=y_true, y_score=y_pred_proba, k=3, normalize=True)

        df_cm = pd.DataFrame(cm, columns=ctg, index=ctg)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
        plt.savefig(result_path + "conf.jpg")

        print(classification_r)
        df = pd.DataFrame([pre, acc, recall, roc_auc, spe, sns, f1_s, top_1_acc, top_2_acc, top_3_acc],
                          index=['pre', 'acc', 'recall', 'roc_auc', 'sp', 'sn', 'f1_score', 'top_1_acc', 'top_2_acc',
                                 'top_3_acc'])

        file_name = f"{ext}_{self.model_name}.xlsx"
        with pd.ExcelWriter(result_path + file_name) as writer:
            df.to_excel(writer, sheet_name='all_metrics')
            classification_r.to_excel(writer, sheet_name='metrics_with_labels')
            worksheet = writer.sheets['all_metrics']
            worksheet.insert_image('E1', result_path + "conf.jpg")
        os.remove(result_path + "conf.jpg")

    def evaluate_f1_score(self):
        val_target, val_preds, val_pred_proba, train_target, train_preds, train_pred_proba = self.evaluate()

        table1 = metrics.classification_report(
            val_target,
            val_preds,
            labels=[int(a) for a in list(self.id2label.keys())],
            target_names=list(self.label2id.keys()),
            zero_division=0,
            digits=self.classification_report_digits,
        )
        print(table1)

        table2 = metrics.classification_report(
            train_target,
            train_preds,
            labels=[int(a) for a in list(self.id2label.keys())],
            target_names=list(self.label2id.keys()),
            zero_division=0,
            digits=self.classification_report_digits,
        )
        print(table2)

        self.__openLogs()
        self.logs_file.write(table1 + "\n")
        self.logs_file.close()

        print("Logs saved at: \033[93m" + self.model_path + "\033[0m")

        return val_target, val_preds, val_pred_proba, train_target, train_preds, train_pred_proba

    def evaluate(self):
        train_preds = []
        train_target = []
        train_pred_proba = []

        val_preds = []
        val_target = []
        val_pred_proba = []

        # For each image
        for image, label in tqdm(self.test):
            val_target.append(label)

            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            preds = outputs.logits.softmax(1).argmax(1).tolist()
            val_preds.extend(preds)

            softmax = torch.nn.Softmax(dim=0)
            pred_soft = softmax(outputs[0][0])
            probabilities = pred_soft.tolist()
            val_pred_proba.append(probabilities)

        for image, label in tqdm(self.train):
            train_target.append(label)

            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            preds = outputs.logits.softmax(1).argmax(1).tolist()
            train_preds.extend(preds)

            softmax = torch.nn.Softmax(dim=0)
            pred_soft = softmax(outputs[0][0])
            probabilities = pred_soft.tolist()
            train_pred_proba.append(probabilities)

        return val_target, val_preds, val_pred_proba, train_target, train_preds, train_pred_proba

    def __openLogs(self):
        self.logs_file = open(self.model_path + "/logs.txt", "a")

    def __getOutputPath(self):

        path = os.path.join(
            self.model_path,
            self.model_name.lower() + "/" + str(self.epoch) + "_" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        )
        if not os.path.isdir(path):
            os.makedirs(path)

        return path

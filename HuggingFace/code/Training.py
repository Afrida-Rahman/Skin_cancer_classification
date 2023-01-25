import os

import PIL
import pandas as pd
import seaborn as sn
import torch
import torchmetrics
from tqdm import tqdm
from datetime import datetime
from sklearn import metrics
from hugsvision.dataio.ImageClassificationCollator import ImageClassificationCollator
from hugsvision.dataio.VisionDataset import VisionDataset
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from transformers import Trainer
from transformers.training_args import TrainingArguments
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np

class Training:

    def __init__(self, train="", test="", id2label={}, label2id={}, model_name="", model_path="", epoch=0, batch=8,
                 model="",
                 feature_extractor="", config="", resolution=224, pretrained_model="", patch=0, lr=1e-4, fp16=True,
                 eval_metric="accuracy", is_best_model=True, test_ratio=0, balanced=False, augmentation=False,
                 save_total_limit=2, weight_decay=0.01, save_steps=10000, evaluation_strategy='epoch',
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
        self.test_ratio = test_ratio
        self.balanced = balanced
        self.augmentation = augmentation
        self.save_total_limit = save_total_limit
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.classification_report_digits = classification_report_digits

    def read_image(self, path):
        data, _, id2label, label2id = VisionDataset.fromImageFolder(
            path,
            test_ratio=self.test_ratio,
            balanced=self.balanced,
            augmentation=self.augmentation,
        )
        return data, _, id2label, label2id

    def build_trainer(self):
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
        self.trainer.train()

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

    def compute_eval_metrics(self, y_true, y_pred, result_path, ext):
        ctg = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        pre = precision_score(y_true=y_true, y_pred=y_pred, average="macro")
        recall = recall_score(y_true=y_true, y_pred=y_pred, average="macro")
        classification_r = pd.DataFrame(classification_report(y_true, y_pred, target_names=ctg, output_dict=True))

        df_cm = pd.DataFrame(cm, columns=ctg, index=ctg)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
        plt.savefig(result_path + "conf.jpg")

        print(classification_r)
        df = pd.DataFrame([pre, acc, recall], index=['pre', 'acc', 'recall'])

        file_name = f"{ext}_{self.model_name}_p{self.patch}_r{self.resolution}_e{self.epoch}.xlsx"
        with pd.ExcelWriter(result_path + file_name) as writer:
            df.to_excel(writer, sheet_name='all_metrics')
            classification_r.to_excel(writer, sheet_name='metrics_with_labels')
            worksheet = writer.sheets['all_metrics']
            worksheet.insert_image('E1', result_path + "conf.jpg")
        os.remove(result_path + "conf.jpg")

    def evaluate_f1_score(self):
        all_target, all_preds = self.evaluate()

        table = metrics.classification_report(
            all_target,
            all_preds,
            labels=[int(a) for a in list(self.id2label.keys())],
            target_names=list(self.label2id.keys()),
            zero_division=0,
            digits=self.classification_report_digits,
        )
        print(table)

        self.__openLogs()
        self.logs_file.write(table + "\n")
        self.logs_file.close()

        print("Logs saved at: \033[93m" + self.model_path + "\033[0m")

        return all_target, all_preds

    def evaluate(self):
        all_preds = []
        all_target = []

        # For each image
        for image, label in tqdm(self.test):
            # Compute
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            # Get predictions from the softmax layer
            preds = outputs.logits.softmax(1).argmax(1).tolist()
            all_preds.extend(preds)

            # Get hypothesis
            all_target.append(label)

        return all_target, all_preds

    def __openLogs(self):
        self.logs_file = open(self.model_path + "/logs.txt", "a")

    def __getOutputPath(self):

        path = os.path.join(
            self.model_path,
            self.model_name.upper() + "/" + str(self.epoch) + "_" + datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        )
        if not os.path.isdir(path):
            os.makedirs(path)

        return path

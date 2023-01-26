from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AutoFeatureExtractor, CvtForImageClassification
from transformers import CvtConfig, CvtModel
import tensorflow as tf

print("GPU: " + str(tf.config.list_physical_devices('GPU')))
print("Available gpu: " + str(tf.test.is_gpu_available()))

absolute_path = "/home/afrida/Documents/skin_cancer_classification/"
train, test, id2label, label2id = VisionDataset.fromImageFolder(
    absolute_path + "raw_data/data_85_15_split/train/",
    test_ratio=0.2,
    balanced=False,
    augmentation=False,
)

configuration = CvtConfig(
    num_channels=3,
    patch_sizes=[7, 3, 3],
    patch_stride=[4, 2, 2],
    patch_padding=[2, 1, 1],
    embed_dim=[64, 192, 384],
    num_heads=[2, 6, 12],
    depth=[2, 8, 16],
    mlp_ratio=[4.0, 4.0, 4.0],
    attention_drop_rate=[0.0, 0.0, 0.0],
    drop_rate=[0.0, 0.0, 0.0],
    drop_path_rate=[0.0, 0.0, 0.1],
    qkv_bias=[True, True, True],
    cls_token=[False, False, True],
    qkv_projection_method=['dw_bn', 'dw_bn', 'dw_bn'],
    kernel_qkv=[3, 3, 3],
    padding_kv=[1, 1, 1],
    stride_kv=[2, 2, 2],
    padding_q=[1, 1, 1],
    stride_q=[1, 1, 1],
    initializer_range=0.02,
    layer_norm_eps=1e-12
)

# configuration = ViTConfig(hidden_size=1024,num_hidden_layers=24, num_attention_heads=16, patch_size=9, image_size=72)

model = CvtModel(configuration)

# configuration = model_85_15_split.config

huggingface_model = 'microsoft/cvt-13'

trainer = VisionClassifierTrainer(
    model_name="Train_Without_Aug_384",
    train=train,
    test=test,
    output_dir=absolute_path + "/HuggingFace/model_85_15_split/",
    max_epochs=100,
    batch_size=8,  # On RTX 2080 Ti
    lr=2e-5,
    fp16=True,
    model=CvtForImageClassification.from_pretrained(huggingface_model,
                                                    num_labels=len(label2id),
                                                    label2id=label2id,
                                                    id2label=id2label,
                                                    ignore_mismatched_sizes=True
                                                    ),
    feature_extractor=AutoFeatureExtractor.from_pretrained("microsoft/cvt-13"),
)
ref, hyp = trainer.evaluate_f1_score()
cm = confusion_matrix(ref, hyp)
labels = list(label2id.keys())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(absolute_path + "HuggingFace/result_85_15_split/conf_no_aug_cvt-13_HAM10k_validation_384.jpg")

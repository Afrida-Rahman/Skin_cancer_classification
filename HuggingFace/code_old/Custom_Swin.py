from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AutoFeatureExtractor, Swinv2ForImageClassification
from transformers import Swinv2Config, Swinv2Model

absolute_path = "/home/afrida/Documents/skin_cancer_classification/"
train, test, id2label, label2id = VisionDataset.fromImageFolder(
    absolute_path + "raw_data/data_85_15_split/train/",
    test_ratio=0.2,
    balanced=False,
    augmentation=False,
)

configuration = Swinv2Config(
    image_size=224,
    patch_size=16,
    num_channels=3,
    embed_dim=64,
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 8, 16],
    window_size=7,
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

# configuration = ViTConfig(hidden_size=1024,num_hidden_layers=24, num_attention_heads=16, patch_size=9, image_size=72)

model = Swinv2Model(configuration)

configuration = model.config

huggingface_model = 'microsoft/swinv2-tiny-patch4-window8-256'

trainer = VisionClassifierTrainer(
    model_name="Train_Without_Aug_checking",
    train=train,
    test=test,
    output_dir=absolute_path + "/HuggingFace/model_85_15_split/",
    max_epochs=5,
    batch_size=32,  # On RTX 2080 Ti
    lr=2e-5,
    fp16=True,
    model=Swinv2ForImageClassification.from_pretrained(huggingface_model,
                                                       label2id=label2id,
                                                       id2label=id2label,
                                                       ignore_mismatched_sizes=True,
                                                       config=configuration
                                                       ),
    feature_extractor=AutoFeatureExtractor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256"),
)
ref, hyp = trainer.evaluate_f1_score()
cm = confusion_matrix(ref, hyp)
labels = list(label2id.keys())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(absolute_path + "HuggingFace/ConfusionMatrix/cm_swinT_without_aug_224.jpg")

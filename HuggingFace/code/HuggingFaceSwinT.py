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
    absolute_path + "raw_data/train_test_splitted/train/",
    test_ratio=0.2,
    balanced=False,
    augmentation=False,
)

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

# configuration = ViTConfig(hidden_size=1024,num_hidden_layers=24, num_attention_heads=16, patch_size=9, image_size=72)

model = Swinv2Model(configuration)

configuration = model.config

huggingface_model = 'microsoft/swinv2-large-patch4-window12-192-22k'

trainer = VisionClassifierTrainer(
    model_name="Train_Without_Aug_192",
    train=train,
    test=test,
    output_dir=absolute_path+ "/HuggingFace/model/",
    max_epochs=50,
    batch_size=8,  # On RTX 2080 Ti
    lr=2e-5,
    fp16=True,
    model=Swinv2ForImageClassification.from_pretrained(huggingface_model,
                                                       num_labels=len(label2id),
                                                       label2id=label2id,
                                                       id2label=id2label,
                                                       ignore_mismatched_sizes=True
                                                       ),
    feature_extractor=AutoFeatureExtractor.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k"),
)
ref, hyp = trainer.evaluate_f1_score()
cm = confusion_matrix(ref, hyp)
labels = list(label2id.keys())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(absolute_path + "HuggingFace/result/conf_no_aug_swin_large_HAM10k_validation_192.jpg")

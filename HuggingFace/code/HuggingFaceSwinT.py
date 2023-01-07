from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AutoFeatureExtractor, SwinForImageClassification, Swinv2ForImageClassification
from transformers import SwinConfig, SwinModel, Swinv2Config, Swinv2Model

train, _, id2label, label2id = VisionDataset.fromImageFolder(
    "../../raw_data/train_test_valid_splitted/train/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

test, _, _, _ = VisionDataset.fromImageFolder(
    "../../raw_data/train_test_valid_splitted/val/",
    test_ratio=0,
    balanced=False,
    augmentation=False,
)

# configuration = Swinv2Config(
#     image_size=192,
#     patch_size=16,
#     num_channels=3,
#     embed_dim=64,
#     depths=[2, 2, 6, 2],
#     num_heads=[2, 4, 8, 16],
#     window_size=8,
#     mlp_ratio=4.0,
#     qkv_bias=True,
#     hidden_dropout_prob=0.0,
#     attention_probs_dropout_prob=0.0,
#     drop_path_rate=0.1,
#     hidden_act='gelu',
#     use_absolute_embeddings=False,
#     patch_norm=True,
#     initializer_range=0.02,
#     layer_norm_eps=1e-05,
#     encoder_stride=32
# )

configuration = SwinConfig()

# configuration = ViTConfig(hidden_size=1024,num_hidden_layers=24, num_attention_heads=16, patch_size=9, image_size=72)

model = SwinModel(configuration)
huggingface_model = 'microsoft/swin-large-patch4-window12-384-in22k'

epoch = 10
model_name = 'Swin'
patch = 4
resolution = 384
window = 12

trainer = VisionClassifierTrainer(
    model_name=f"{model_name}_p{patch}_w{window}_r{resolution}_e{epoch}",
    train=train,
    test=test,
    output_dir="../model/swin/",
    max_epochs=10,
    batch_size=4,  # On RTX 2080 Ti
    lr=2e-5,
    fp16=False,
    model=SwinForImageClassification.from_pretrained(huggingface_model,
                                                       num_labels=len(label2id),
                                                       label2id=label2id,
                                                       id2label=id2label,
                                                       ignore_mismatched_sizes=True
                                                       ),
    feature_extractor=AutoFeatureExtractor.from_pretrained(huggingface_model),
)
ref, hyp = trainer.evaluate_f1_score()
cm = confusion_matrix(ref, hyp)
labels = list(label2id.keys())
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt="")
plt.savefig(f"../result/swin/val_conf_{model_name}_p{patch}_w{window}_r{resolution}_e{epoch}.jpg")
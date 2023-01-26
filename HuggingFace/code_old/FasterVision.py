# Link: https://exchange.scale.com/public/blogs/faster-vision-transformers-for-supervised-image-classification
from keras import Input
from keras.datasets import fashion_mnist
from keras.layers import Dense, MultiHeadAttention, LayerNormalization, Layer
from keras.layers import Embedding, GlobalAveragePooling1D
from keras.layers import Conv2D, Dropout
from tensorflow import reduce_mean, float32, range, reshape
from keras import utils
from keras.metrics import TopKCategoricalAccuracy
from tensorflow.nn import gelu
from keras import Model, Sequential
from tensorflow.image import extract_patches
import matplotlib.pyplot as plt
import time
import numpy as np


class ImageTokenizerLayer(Layer):
    def __init__(self, token_shape):
        super(ImageTokenizerLayer, self).__init__()
        self.token_shape = token_shape

    def call(self, images):
        tokens = extract_patches(
            images=images,
            sizes=[1, self.token_shape[0], self.token_shape[1], 1],
            strides=[1, self.token_shape[0], self.token_shape[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID", )
        return tokens


def display_tokens(img, tokenized_img):  # Check if RGB or grayscale image
    if img.shape[-1] == 3:
        img = img[:, :, :]
    else:
        img = img[:, :, 0]
    plt.imshow(img.astype("uint8"))
    plt.title('Train Image')
    fig, ax = plt.subplots(nrows=tokenized_img.shape[0], ncols=tokenized_img.shape[1], figsize=(4, 4),
                           subplot_kw=dict(xticks=[], yticks=[]))

    for i in range(tokenized_img.shape[0]):
        for j in range(tokenized_img.shape[1]):  # Check if it an RGB image
            if tokenized_img.shape[-1] == 3:
                token = np.reshape(tokenized_img[i, j, :], (token_dims[0], token_dims[1], 3))
            else:  # Show graysclae
                token = np.reshape(tokenized_img[i, j, :], (token_dims[0], token_dims[1]))

    ax[i, j].imshow(token)
    plt.gcf().suptitle('Image Tokens')
    fig, ax = plt.subplots(nrows=1, ncols=tokenized_img.shape[0] * tokenized_img.shape[1], figsize=(18, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))

    for i in range(tokenized_img.shape[0]):
        for j in range(tokenized_img.shape[1]):  # Check if it an RGB image
            if tokenized_img.shape[-1] == 3:
                token = np.reshape(tokenized_img[i, j, :], (token_dims[0], token_dims[1], 3))
            else:  # Show graysclae
                token = np.reshape(tokenized_img[i, j, :], (token_dims[0], token_dims[1]))

    ax[i * tokenized_img.shape[1] + j].imshow(token)
    plt.gcf().suptitle('Sequence of image tokens')
    # plt.show()  # Take the square root of dimensions of the input image


class ImageEmbeddingLayer(Layer):
    def __init__(self, output_dim):
        super(ImageEmbeddingLayer, self).__init__()
        self.output_dim = output_dim
        # Need to define the Dense layer for linear projections # Embedding layer for positions

    def build(self, input_shape):
        self.total_img_tokens = input_shape[1] * input_shape[2]
        self.token_dims = input_shape[3]
        self.normalize_layer = LayerNormalization()
        self.dense = Dense(units=self.output_dim, input_shape=(None, self.token_dims))
        self.position_embedding = Embedding(input_dim=self.total_img_tokens, output_dim=self.output_dim)

    def call(self, input):
        img_tokens = reshape(input, [-1, self.total_img_tokens, self.token_dims])
        normalized_img_token = self.normalize_layer(img_tokens)
        img_projection = self.dense(normalized_img_token)
        all_positions = range(start=0, limit=self.total_img_tokens, delta=1)
        positions_encoding = self.position_embedding(all_positions)
        return positions_encoding + img_projection


class EncoderLayer(Layer):
    def __init__(self, total_heads, total_dense_units, embed_dim):
        super(EncoderLayer, self).__init__()  # Multihead attention layer
        self.multihead = MultiHeadAttention(num_heads=total_heads, key_dim=embed_dim)  # Feed forward network layer
        self.nnw = Sequential([Dense(total_dense_units, activation="relu"), Dense(embed_dim)])  # Normalization
        self.normalize_layer = LayerNormalization()

    def call(self, inputs):
        attn_output = self.multihead(inputs, inputs)
        normalize_attn = self.normalize_layer(inputs + attn_output)
        nnw_output = self.nnw(normalize_attn)
        final_output = self.normalize_layer(normalize_attn + nnw_output)
        return final_output


# Set the hyperparameters
n_classes = 7
EMBED_DIM = 32
NUM_HEADS = 7
TOTAL_DENSE = 100
EPOCHS = 50
FINAL_DENSE = 150
DROPOUT = 0.01


def build_vit(input_shape, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
              total_dense_units=TOTAL_DENSE):  # Start connecting layers
    inputs = Input(shape=input_shape)
    embedding_layer = ImageEmbeddingLayer(embed_dim)(inputs)
    encoder_layer1 = EncoderLayer(num_heads, total_dense_units, embed_dim)(embedding_layer)
    encoder_layer2 = EncoderLayer(num_heads, total_dense_units, embed_dim)(encoder_layer1)
    encoder_layer3 = EncoderLayer(num_heads, total_dense_units, embed_dim)(encoder_layer2)
    encoder_layer4 = EncoderLayer(num_heads, total_dense_units, embed_dim)(encoder_layer3)
    pooling_layer = GlobalAveragePooling1D()(encoder_layer4)
    dense_layer = Dense(FINAL_DENSE, activation='relu')(pooling_layer)
    dropout_layer = Dropout(DROPOUT)(dense_layer)
    outputs = Dense(n_classes, activation="softmax")(dense_layer)  # Construct the transformer model_85_15_split
    ViT = Model(inputs=inputs, outputs=outputs)
    ViT.compile(optimizer="adam", loss='categorical_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    return ViT


root = "/home/afrida/Documents/skin_cancer_classification/raw_data/"
train_img = np.load(root + "train_img_28x28.npy")
train_label = np.load(root + "train_label_ctg.npy")
test_img = np.load(root + "test_img_28x28.npy")
test_label = np.load(root + "test_label_ctg.npy")

print(f"Training Data: x_train: {train_img.shape} - y_train: {train_label.shape}")
print(f"Test Data: x_test {test_img.shape} - y_test: {test_label.shape}")

token_dims = np.round(np.sqrt(train_img[0].shape)).astype("uint8")
train_tokens = ImageTokenizerLayer(token_dims)(train_img)
test_tokens = ImageTokenizerLayer(token_dims)(test_img)
print('Train tokens shape', train_tokens.shape)
print('Test tokens shape:', test_tokens.shape)


vit = build_vit(train_tokens[0].shape)

history = vit.fit(train_tokens, train_label, batch_size=32, epochs=EPOCHS, validation_split=0.2)
end_time = time.time()

print("Results:")
print('Test label shape', test_label.shape)
print('Test tokens shape:', test_tokens.shape)
train_metrics = vit.evaluate(train_tokens, train_label)
test_metrics = vit.evaluate(test_tokens, test_label)


n_metrics = int(len(history.history.keys())/2)
metrics = list(history.history.values())
metric_names = list(history.history.keys())

print('\nTraining set evaluation of ViT')
for i, value in enumerate(train_metrics):
  print(metric_names[i], ': ', value)

print('\nTest set evaluation of ViT')
for i, value in enumerate(test_metrics):
  print(metric_names[i], ': ', value)

fig = plt.figure(figsize=(18,4))
for i in np.arange(n_metrics):
  fig.add_subplot(101 + n_metrics*10 +i)
  plt.plot(metrics[i])
  plt.plot(metrics[i+n_metrics])
  plt.legend(['Training', 'Validation'])
  plt.xlabel('Epoch Number')
  plt.ylabel(metric_names[i])

  plt.gcf().suptitle('Learning History of the ViT')
  plt.subplots_adjust(wspace=0.4)
plt.savefig(f"./metrics_plot.png")

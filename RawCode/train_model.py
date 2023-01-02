import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

num_classes = 7
input_shape = (72, 72, 3)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 30
image_size = 28
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 7
transformer_units = [projection_dim * 2, projection_dim, ]
transformer_layers = 8
mlp_head_units = [2048, 1024]


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    # x0 = tf.keras.layers.Flatten()(logits)
    # x1 = tf.keras.layers.BatchNormalization()(x0)
    # x2 = tf.keras.layers.Dense(11, activation='relu')(x1)
    # x3 = tf.keras.layers.BatchNormalization()(x2)
    # outputs = tf.keras.layers.Dense(7, activation='softmax')(x2)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    checkpoint_filepath = "../checkpoint/"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=train_img,
        y=train_label,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )
    return history, model


root = "raw_data/"
train_img = np.load(root + "train_img_72x72.npy")
train_label = np.load(root + "train_label_72x72.npy")
test_img = np.load(root + "test_img_72x72.npy")
test_label = np.load(root + "test_label_72x72.npy")

print(f"x_train: {train_img.shape} - y_train: {train_label.shape}")
print(f"x_test {test_img.shape} - y_test: {test_label.shape}")
# data_augmentation = keras.Sequential(
#     [
#         layers.Normalization(),
#         layers.Resizing(image_size, image_size),
#         layers.RandomFlip("horizontal"),
#         layers.RandomRotation(factor=0.02),
#         layers.RandomZoom(height_factor=0.2, width_factor=0.2),
#     ], name="data_augmentation", )
# data_augmentation.layers[0].adapt(train_img)

vit_classifier = create_vit_classifier()
history, model = run_experiment(vit_classifier)

model.load_weights("./checkpoint/")
_, accuracy = model.evaluate(test_img, test_label)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

pred_img= model.predict(test_img)
pred_label = np.argmax(pred_img, axis=1)

cf_matrix = confusion_matrix(test_label, pred_label)


import seaborn as sns

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues') #/np.sum(cf_matrix), , fmt='.2%'

ax.set_title('Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

ax.xaxis.set_ticklabels(['akiec','bcc','bkl','df','mel','vasc','nv'])
ax.yaxis.set_ticklabels(['akiec','bcc','bkl','df','mel','vasc','nv'])

plt.show()



import tensorflow as tf
import sys
import numpy as np
from numpy.random import seed

sys.path.append('/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/Swin-Transformer-TF')
from swintransformer import SwinTransformer

lr = .01
epoch = 10
sd = 203
drp = 0
model_name = "small_224"
seed(sd)
tf.random.set_seed(sd)

print("GPU: " + str(tf.config.list_physical_devices('GPU')))
print("Available gpu: " + str(tf.test.is_gpu_available()))

data_path = "/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/customized_data/"
result_path = "/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/result/"
model_path = "/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/model/"

x_train = np.load(data_path+ "x_train.npy")
y_train = np.load(data_path+ "y_train.npy")

IMAGE_SIZE = [x_train.shape[1], x_train.shape[1], 3]

swin_vit = SwinTransformer('swin_'+model_name, include_top=True, pretrained=True)


# checkpoint_path = "/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/model/small_200e_115sd_0.0001l"
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=IMAGE_SIZE))
# convLayer = tf.keras.layers.Conv2D(7, 3, padding= 'same', activation='relu',input_shape=IMAGE_SIZE) (x_train)
# print(convLayer.shape)
# model.add(convLayer)
# model.add(tf.keras.layers.Conv2D(7, 3, padding= 'same', activation='relu',input_shape=IMAGE_SIZE))
# model.add(tf.keras.layers.MaxPooling2D())
model.add(swin_vit)
model.add(tf.keras.layers.Dense(7, activation='softmax'))
model.build(IMAGE_SIZE)
# model = tf.keras.models.load_model(checkpoint_path)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_path = f"/home/afrida/Documents/skin_cancer_classification/breaSwinTNet/checkpoint/best_weights_{model_name}.hdf5"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
model.fit(x_train, y_train, batch_size=2, epochs=epoch, validation_split=0.20, callbacks=[cp_callback])

model.save(model_path + f"{model_name}_{epoch}e_{sd}sd_{lr}lr_{drp}drp")

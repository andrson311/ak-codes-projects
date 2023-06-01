import tensorflow as tf
import numpy as np
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import h5py
import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

ROOT = os.path.dirname(__file__)

api.dataset_download_files(
    'daavoo/3d-mnist',
    path=os.path.join(ROOT, 'data'),
    unzip=True
)

# hyperparameters
BATCH_SIZE = 128
EPOCHS = 100
CLASSES = 10


def array_to_color(array, cmap='Oranges'):
    s_m = ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]

def rgb_data_transform(data):
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
    return np.asarray(data_t, dtype=np.float32)

with h5py.File(os.path.join(ROOT, 'data', 'full_dataset_vectors.h5'), 'r') as hf:
    X_train = hf['X_train'][:]
    y_train = hf['y_train'][:]
    X_test = hf['X_test'][:]
    y_test = hf['y_test'][:]

    sample_shape = (16, 16, 16, 3)
    X_train = rgb_data_transform(X_train)
    X_test = rgb_data_transform(X_test)

    y_train = tf.keras.utils.to_categorical(y_train).astype(np.integer)
    y_test = tf.keras.utils.to_categorical(y_test).astype(np.integer)

model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
    tf.keras.layers.Dense(CLASSES, activation='softmax')
])

print(model.summary())

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ROOT, 'training_checkpoints', 'ckpt_{epoch}'),
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True),
]

history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=callbacks)

score = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# Plot history: Categorical crossentropy & Accuracy
plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
plt.plot(history.history['accuracy'], label='Accuracy (training data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Model performance for 3D MNIST Keras Conv3D example')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
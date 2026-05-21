import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix

train_ds = image_dataset_from_directory(
    directory='C:/Users/student/Desktop/LV9_Zavođa/gtsrb/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset="training",
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

validation_ds = image_dataset_from_directory(
    directory='C:/Users/student/Desktop/LV9_Zavođa/gtsrb/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset="validation",
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)

test_ds = image_dataset_from_directory(
    directory='C:/Users/student/Desktop/LV9_Zavođa/gtsrb/Test',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48),
    shuffle=False
)

model = keras.Sequential()
model.add(layers.Input(shape=(48, 48, 3)))
model.add(layers.Rescaling(1./255))

filter_sizes = [32, 64, 128]
for x in filter_sizes:
    model.add(layers.Conv2D(filters=x, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=x, kernel_size=(3, 3), strides=1, padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Dropout(rate=0.2))

model.add(layers.Flatten())
model.add(layers.Dense(units=512, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(units=43, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='logs/',
    histogram_freq=1
)

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=10,
    callbacks=[checkpoint_callback, tensorboard_callback]
)

best_model = keras.models.load_model('best_model.keras')
loss, accuracy = best_model.evaluate(test_ds)
print(f"Točnost: {accuracy * 100:.2f}%")

y_true = []
y_pred = []
for images, labels in test_ds:
    preds = best_model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
print(cm)

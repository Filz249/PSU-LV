import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('best_model.keras')

img = keras.utils.load_img('putanja_do_slike.jpg', target_size=(48, 48))
img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predvidena_klasa = np.argmax(predictions, axis=1)[0]
vjerojatnost = predictions[0][predvidena_klasa]

print(f"Klasa: {predvidena_klasa}")
print(f"Vjerojatnost: {vjerojatnost * 100:.2f}%")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

image_size = (180, 180)

# Load trained checkpoint
model = keras.models.load_model("save_at_2.keras")

# Load image
img = keras.utils.load_img(
    "PetImages/Cat/6779.jpg",
    target_size=image_size
)

plt.imshow(img)

# Visualize the image for user
plt.show(block=False)
plt.pause(2)
plt.close()

# Convert to array
img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Add batch dimension

# Predict
predictions = model.predict(img_array)

# Since model was trained with from_logits=True
score = float(keras.ops.sigmoid(predictions[0][0]))

print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")

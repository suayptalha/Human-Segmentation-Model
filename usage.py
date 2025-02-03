from tensorflow.keras.models import load_model

model = load_model("seg-model.h5")

import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

test_image_path = "image_path"
input_image = preprocess_image(test_image_path)

predicted_mask = model.predict(input_image)[0]
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.imread(test_image_path)[:, :, ::-1])
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title("Segmentation Mask")

plt.show()

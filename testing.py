import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model

model = load_model('HasilTraining.h5')

class_labels = ["2", "3", "5", "7", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
                "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]


def preprocess_image(img):
    img = cv2.resize(img, (90, 100))
    img_array = img.astype(np.float32) / 255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_array = preprocess_image(gray_img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    plt.imshow(gray_img, cmap='gray')
    plt.axis('off')
    plt.show()

    print('Predicted Class:', predicted_label)

input_image = cv2.imread('/content/test 1.png')
predict_image(model, input_image)
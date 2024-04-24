import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

train = ImageDataGenerator(rescale= 1/255,
                           brightness_range=(0.8, 1.2),
                           zoom_range=[0.8, 1.2],
                           rotation_range=27)
validation = ImageDataGenerator(rescale= 1/255)


train_dataset = train.flow_from_directory("dataset/training", target_size=(100, 90), batch_size = 6, class_mode = 'categorical', color_mode="grayscale")
validation_dataset = validation.flow_from_directory("dataset/validation",target_size= (100, 90), batch_size = 6, class_mode = 'categorical', color_mode='grayscale')



model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(100, 90, 1)),
                                    tf.keras.layers.MaxPooling2D(),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(len(train_dataset.class_indices), activation='softmax')])


model.compile(loss='categorical_crossentropy', optimizer=Adam(),
              metrics=['accuracy'])
print(model.summary())


model_fit = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)
model.save('HasilTraining.h5')

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(model_fit.history['loss'], label='Training Loss')
plt.plot(model_fit.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(model_fit.history['accuracy'], label='Training Accuracy')
plt.plot(model_fit.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

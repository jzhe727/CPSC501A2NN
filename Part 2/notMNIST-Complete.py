#Original Author: Jonathan Hudson
#CPSC 501 F22

import tensorflow as tf
import numpy as np

tf.random.set_seed(1234)

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("--Process data--")
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
input_shape = (28, 28, 1)

print("--Make model--")
model = tf.keras.models.Sequential(
  [
  tf.keras.layers.Conv2D(8, (5, 5), padding = 'same', input_shape = (28, 28, 1)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(8, (3, 3), padding = 'same'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.2,
    decay_steps=1875,
    decay_rate=0.75,
    staircase=True)

sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("--Fit model--")
model.fit(x_train, y_train, epochs=10, verbose=2)

print("--Evaluate model--")
model_loss1, model_acc1 = model.evaluate(x_train,  y_train, verbose=2)
model_loss2, model_acc2 = model.evaluate(x_test,  y_test, verbose=2)
print(f"Train / Test Accuracy: {model_acc1*100:.1f}% / {model_acc2*100:.1f}%")

model.save('notMNIST.h5')
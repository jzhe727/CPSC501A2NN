import tensorflow as tf

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#read data
data_in = pd.read_csv('heart.csv')
data_in.pop('row.names')

data_in['sbp'] = data_in['sbp'].astype(np.float32)
data_in['typea'] = data_in['typea'].astype(np.float32)
data_in['age'] = data_in['age'].astype(np.float32)
data_in['chd'] = data_in['chd'].astype(np.float32)

def convert(name):
    if name == 'Present':
        return 1
    else:
        return 0

data_in['famhist'] = data_in['famhist'].apply(convert)

TOTAL = 462 #I split off the last 1/7 of the data as test data
train = data_in.head(TOTAL-66)
test = data_in.tail(66)

train.to_csv('heart_train.csv')
test.to_csv('heart_test.csv')

training_y = train.pop('chd')
training_x = train

testing_y = test.pop('chd')
testing_x = test

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(training_x)
model = tf.keras.models.Sequential(
  [
  normalizer,
  tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.005)),
  tf.keras.layers.Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(l2=0.001)),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='bce', optimizer='adam', metrics = ['accuracy'])
model.fit(training_x, training_y, batch_size = 1, epochs=30, verbose=2)

model_loss1, model_acc1 = model.evaluate(training_x,  training_y, verbose=2)
model_loss2, model_acc2 = model.evaluate(testing_x,  testing_y, verbose=2)
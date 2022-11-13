import tensorflow as tf
import pandas as pd

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#read data
data_in = pd.read_csv('data-input.csv')

data_in.to_csv('data.csv')

#clean data and output
data_clean = data_in.drop(['id', 'zipcode'], axis = 1)
ordering = data_clean.columns.values.tolist()
ordering[0], ordering[1] = ordering[1], ordering[0]
data_clean = data_clean.reindex(ordering, axis = 1)
data_clean['date'] = data_clean['date'].str.slice(0, 8)
data_clean.to_csv('data-cleaned.csv')

#visualize data
fig1, ax1 = plt.subplots(1, 3, sharey = True)
fig1.suptitle('Histograms of Some Housing Attributes')
sns.histplot(data_clean['bedrooms'], ax = ax1[0]) #[15870] claims 33 bedrooms in the dataset, might be inaccurate
ax1[0].set_title('Bedrooms')
sns.histplot(data_clean['bathrooms'], ax = ax1[1])
ax1[1].set_title('Bathrooms')
sns.histplot(data_clean['floors'], ax = ax1[2])
ax1[2].set_title('Floors')
plt.savefig('plot1.png')
plt.clf()

data2 = sns.scatterplot(data=data_clean, x = 'long', y = 'lat')
data2.set_title('House Sale Locations')
plt.savefig('plot2.png')
plt.clf()

data3 = sns.histplot(data= data_clean['price'])
data3.set_title('Histogram of House Price (in millions)')
plt.savefig('plot3.png')
plt.clf()

# With help from tutorial https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

data_nn = pd.read_csv('data-cleaned.csv')
target = data_nn.pop('price')
data_nn.pop('Unnamed: 0')
# 21613 datapoints, use ~20% for testing

# All features are numeric types, as the categorical inputs (like the boolean waterfront) are already preprocessed into
# integer representations.
TOTAL = 21613
training_x = data_nn.head(TOTAL-4000)
training_y = target.head(TOTAL-4000)

testing_x = data_nn.tail(4000)
testing_y = target.tail(4000)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(training_x)

model = tf.keras.models.Sequential(
  [
  normalizer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])

# using mean squared error for price prediction
sgd = tf.keras.optimizers.SGD(learning_rate= (10**(-9)))
model.compile(optimizer=sgd, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

model.fit(training_x, training_y, batch_size = 32, epochs=10, verbose=2)

model_loss1, model_acc1 = model.evaluate(training_x,  training_y, verbose=2)
model_loss2, model_acc2 = model.evaluate(testing_x,  testing_y, verbose=2)
print(f"Train / Test Accuracy: {model_acc1:.1f} mse / {model_acc2:.1f} mse")
print(f"Train / Test Error: {model_acc1**0.5:.1f} var / {model_acc2**0.5:.1f} var")

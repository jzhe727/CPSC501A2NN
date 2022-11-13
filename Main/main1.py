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

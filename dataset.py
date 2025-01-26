import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import os
from glob import glob

csv = 'results.csv'
dataset = pd.read_csv(csv)
dataset['male'] = dataset['gender'].map(lambda x: "1" if x == 'M' else "0")
dataset['path'] = dataset['name'].map(lambda x: "test22/"+str(x))
dataset['exists'] = 'True'
print(dataset)

print(str(sum([1 if i else 0 for i in dataset['exists']]))+" images found out of "+str(len(dataset['exists']))+" images.")

dataset[['age','male']].hist(figsize = (10, 5))

age_groups = 4

dataset['age_range'] = pd.qcut(dataset['age'],age_groups)
sample_dataset = dataset.groupby(['age','male']).apply(lambda x: x.sample(1)).reset_index(drop=True)

fig,m_axes = plt.subplots(age_groups,2)
for c_ax, (_,c_row) in zip(m_axes.flatten(),sample_dataset.sort_values(['age_range','gender']).iterrows()):
	c_ax.imshow(imread(c_row['path']))
	c_ax.axis('off')
	c_ax.set_title('{age} years, {gender}'.format(**c_row))
plt.show()
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import os
from glob import glob

csv = 'agesex_all_combined.csv'
dataset = pd.read_csv(csv)
dataset['sex'] = dataset['male'].map(lambda x: "1" if x == 'True' else "0")
dataset['exists'] = 'True'

print(str(sum([1 if i else 0 for i in dataset['exists']]))+" images found out of "+str(len(dataset['exists']))+" images.")

dataset[['age','sex']].hist(figsize = (5, 5), bins=10)
age_groups = 4
dataset['age_range'] = pd.qcut(dataset['age'],age_groups)

print(dataset)
sample_dataset = dataset.groupby(['age','sex']).apply(lambda x: x.sample(1)).reset_index(drop=True)

#fig,m_axes = plt.subplots(age_groups,2)
# for c_ax, (_,c_row) in zip(m_axes.flatten(),sample_dataset.sort_values(['age_range','sex']).iterrows()):
	# c_ax.axis('off')
	# c_ax.set_title('{age} years, {sex}'.format(**c_row))
plt.show()
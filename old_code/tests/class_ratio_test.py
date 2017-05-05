from data_loader import DataLoader
import numpy as np

data_loader = DataLoader('imdb')
data = data_loader.load_dataset()
labels = list(data.values())
num_labels = len(labels)
num_men = np.sum(labels)
num_women = num_labels - num_men
print('Number of men:', num_men)
print('Number of women:', num_women)

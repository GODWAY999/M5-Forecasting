from function.DataProcess import load_data, data_split
from model.CrossValidation import CV
import numpy as np


data, label = data_split(load_data('../data/train.csv'))
print(label.shape)
print(data.shape)
model = CV()
model.train(data, label, 10)


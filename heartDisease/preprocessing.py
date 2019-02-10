import pandas as pd
import numpy as np
import csv

data = pd.read_csv('heart.csv')


X = data.iloc[:, :13]
Y = data.iloc[:, 13]


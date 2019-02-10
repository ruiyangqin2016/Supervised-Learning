import csv
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_csv('winequality-white.csv')
print('Correlation: ', np.corrcoef(data.iloc[:,0], data.iloc[:,1])[0][1])

print('Correlation between fixed.acidity and volatile.acidity: ', np.corrcoef(data.iloc[:,0], data.iloc[:,1])[0][1])
print('Correlation between fixed.acidity and citric.acid:      ', np.corrcoef(data.iloc[:,0], data.iloc[:,2])[0][1])
print('Correlation between volatile.acidity and citric.acid:   ', np.corrcoef(data.iloc[:,1], data.iloc[:,2])[0][1])
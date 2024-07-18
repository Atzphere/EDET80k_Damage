import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DPATH = "./gaussian profile 1 2.5A 60S.csv"

data = pd.read_csv(DPATH, header=None)

time = np.cumsum(data[2]) - data[2][0]
voltage = data[3]

print(data)
plt.plot(time, voltage)
plt.show()

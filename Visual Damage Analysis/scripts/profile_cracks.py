import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

DPATH = "..\\data\\shadows\\"

files = ["onehour.png"]

i = files[0]

icx = 918
icy = 1053

radius = 1172 # pixels

# fig, ax = plt.subplots()
img = cv2.imread(DPATH + i)
loaded_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ax.imshow(255 - loaded_image, cmap='Greys')
blur = 255 - cv2.blur(loaded_image,(25,25))
# plt.show()

coord_y, coord_x = np.indices(np.shape(blur))

coord_y = coord_y - 1078
coord_x = coord_x - 1278

radii = np.sqrt(coord_y**2 + coord_x**2)

intensities = []

for r in tqdm(np.arange(500, radius, 1)):
    ring = blur[np.abs(radii - r) > 1.41]
    intensities.append(np.mean(ring))

plt.plot(intensities)
plt.show()

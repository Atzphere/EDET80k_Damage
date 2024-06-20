import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DPATH = "..\\data\\damage photos\\"

files = ["Image-FreeModeAcquisition-01--01.png", "Image-FreeModeAcquisition-01--02.png",
         "Image-FreeModeAcquisition-01--03.png", "Image-FreeModeAcquisition-01--04.png"]

for i in files:
    loaded_image = np.array(Image.open(DPATH + i).convert('L'))
    plt.imshow(255 - loaded_image, cmap='Greys', alpha=0.4)
plt.show()

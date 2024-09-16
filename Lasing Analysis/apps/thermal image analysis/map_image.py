import temperaturemap
import perspectivemap
import numpy as np
import optrisimagelib as oil
import matplotlib.pyplot as plt
import cv2


AlASB = temperaturemap.maps["Al"]
print(AlASB.true_temperature(27.6))
IMDIMS = (382, 288)
CHIP_BOTTOMLEFT = (151, 274)
CHIP_TOPRIGHT = (315, 109)

mintemp = 10
maxtemp = 70


im = oil.load_optris_imcsv("test2.csv")
transform = perspectivemap.image_transform
converted_im = cv2.warpPerspective(
    im, transform, IMDIMS, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
coords_y, coords_x = np.indices(np.shape(converted_im))


fig, ax = plt.subplots(2)
ax[0].imshow(converted_im, cmap="magma", vmin=mintemp, vmax=maxtemp)
ax[0].scatter(*CHIP_TOPRIGHT, s=2)
ax[0].scatter(*CHIP_BOTTOMLEFT, s=2)
# np.logical_and(coords_y <= CHIP_BOTTOMLEFT[1], coords_y >= CHIP_TOPRIGHT[1])
# cy = coords_y[~np.logical_and(coords_y <= CHIP_BOTTOMLEFT[1], coords_y >= CHIP_TOPRIGHT[1])]
# cx = coords_x[~np.logical_and(coords_x >= CHIP_BOTTOMLEFT[0], coords_x <= CHIP_TOPRIGHT[0])]
# print(cy)
converted_im[109:275,151:316] = AlASB.true_temperature(converted_im[109:275,151:316])
print('e')
ax[1].imshow(converted_im, cmap="magma", vmin=mintemp, vmax=maxtemp)
plt.show()

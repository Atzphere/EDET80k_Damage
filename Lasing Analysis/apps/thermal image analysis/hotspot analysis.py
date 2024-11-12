import temperaturemap
import perspectivemap
import numpy as np
import optrisimagelib as oil
import matplotlib.pyplot as plt
import cv2
import time
import dirtools as dt
import matplotlib_animtools as man
from tqdm import tqdm

FPATH = r"F:\PIX Connect Recordings (2024+)\framewise profiling"

files = list(dt.get_files(FPATH, fullpath=True))

AlASB = temperaturemap.maps["Al"]
print(AlASB.true_temperature(27.6))
IMDIMS = (382, 288)
CHIP_BOTTOMLEFT = (156, 273)
CHIP_TOPRIGHT = (317, 115)

spot_bl = (269, 158)
bl_x, bl_y = spot_bl
spot_tr = (290, 135)
tr_x, tr_y = spot_tr

mintemp = 10
maxtemp = 70

# disp1 = ax[0].imshow(np.zeros((15, 15)), cmap="magma", vmin=mintemp, vmax=maxtemp)
# disp2 = ax[1].imshow(np.zeros((15, 15)), cmap="magma", vmin=mintemp, vmax=maxtemp)

arrs1 = []
arrs2 = []

for num, f in tqdm(enumerate(files)):
    try:
        im = oil.load_optris_imcsv(f)
    except UnicodeDecodeError:
        continue
    transform = perspectivemap.image_transform
    converted_im = cv2.warpPerspective(
        im, transform, IMDIMS, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coords_y, coords_x = np.indices(np.shape(converted_im))

    roi = converted_im[tr_y:bl_y, bl_x:tr_x]
    conv_roi = AlASB.true_temperature(roi)

    arrs1.append(converted_im)
    arrs2.append(conv_roi)

grads = []
for arr in arrs1:
    fx, fy = np.gradient(arr)
    grads.append(np.sqrt(fx**2 + fy**2))
anim = man.animate_2d_arrays(grads, cmap="magma", vmin=0, vmax=0.5, interval=0.037 * 1000)
plt.show()




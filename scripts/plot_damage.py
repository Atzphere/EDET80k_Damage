import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import numpy as np
import pandas as pd

DPATH = "../data/EDET80k ROI - MachineReadable.csv"
EPATH = "../export/"

sns.set_theme()

DSIZE_DEFAULT = 5

DSIZE_HUGE = 32
DSIZE_LARGE = 16
DSIZE_MEDIUM = 8
DSIZE_SMALL = 4
DSIZE_TINY = 2


data_references = {"huge": DSIZE_HUGE, "large": DSIZE_LARGE,
                   "medium": DSIZE_MEDIUM, "small": DSIZE_SMALL, "tiny": DSIZE_TINY}

data = pd.read_csv(DPATH, header=0)


def get_size(text):
    for word in text.split():
        if word in data_references.keys():
            return data_references[word]**2
    return DSIZE_DEFAULT**2


fig, ax = plt.subplots()
ax.scatter(data["chipX"], data["chipY"], s=list(map(get_size, data["ROI"])), label="Baseline", alpha=0.5)
rect = patches.Rectangle((0, 6), 32, 32, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)

ax.set_xlim(0, 49)
ax.set_ylim(0, 38)
ax.set_aspect('equal', adjustable='box')

ax.set_title("Damaged spots on EDET80k dummy chip")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")

ax.legend()

plt.savefig(EPATH+"for_presentation.png", dpi='figure', format="png")

plt.show()
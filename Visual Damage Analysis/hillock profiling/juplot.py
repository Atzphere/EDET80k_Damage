'''
Matplotlib wrapper for ipynb notebooks; does garbage cleanup on cells with plots when they're re-run.
'''

import matplotlib.pyplot as plt

def juplot(num, *args, **kwargs):
    plt.close(num)
    fig, ax = plt.subplots(*args, num=num, **kwargs)
    return fig, ax
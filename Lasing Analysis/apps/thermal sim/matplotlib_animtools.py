import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_2d_arrays(arrays, interval=200, repeat=True, cmap='viridis', repeat_delay=0, vmin=0, vmax=255):
    """
    Animates an arbitrary array of 2D arrays using plt.imshow.
    
    Parameters:
    - arrays: list of 2D arrays to animate
    - interval: delay between frames in milliseconds (default is 200ms)
    - repeat: whether the animation should repeat when the sequence is completed (default is True)
    - cmap: colormap to be used in imshow (default is 'viridis')
    """
    fig, ax = plt.subplots()
    im = ax.imshow(arrays[0], cmap=cmap, interpolation="gaussian", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)

    def update(frame):
        im.set_array(arrays[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames=len(arrays), interval=interval, repeat=repeat, repeat_delay = repeat_delay)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Create some example data: a list of 2D arrays
    data = [np.random.rand(10, 10) for _ in range(30)]
    
    # Call the function to animate the data
    animate_2d_arrays(data)

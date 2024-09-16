import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_2d_arrays(arrays, interval=200, repeat=True, cmap='viridis', repeat_delay=0, vmin=0, vmax=255, push=False):
    """
    Animates an arbitrary array of 2D arrays using plt.imshow.

    Parameters:
    - arrays: list of 2D arrays to animate
    - interval: delay between frames in milliseconds (default is 200ms)
    - repeat: whether the animation should repeat when the sequence is completed (default is True)
    - cmap: colormap to be used in imshow (default is 'viridis')
    """
    fig, ax = plt.subplots()
    im = ax.imshow(arrays[0], cmap=cmap,
                   interpolation="gaussian", vmin=vmin, vmax=vmax)
    fig.colorbar(im)

    def update(frame):
        im.set_array(arrays[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames=len(
        arrays), interval=interval, repeat=repeat, repeat_delay=repeat_delay)
    if push:
        plt.show()
    else:
        return ani


def animate_1d_arrays(x, arrays, interval=200, repeat=True, repeat_delay=0, push=False):
    """
    Animates an arbitrary array of 1D arrays using plt.plot.

    Parameters:
    - x: array of x values.
    - arrays: list of 1D arrays to plot as a function of x.
    - interval: delay between frames in milliseconds (default is 200ms)
    - repeat: whether the animation should repeat when the sequence is completed (default is True)
    - cmap: colormap to be used in imshow (default is 'viridis')
    """
    fig, ax = plt.subplots()
    im = ax.plot(x, arrays[0])
    # ax.set_ylim(bottom=0, top=1200)

    def update(frame):
        im[0].set_data(x, arrays[frame])
        return [im]

    ani = FuncAnimation(fig, update, frames=len(
        arrays), interval=interval, repeat=repeat, repeat_delay=repeat_delay)

    if push:
        plt.show()
    else:
        return ani


def animate_1d_lines(x, arrays, labels, interval=200, repeat=True, repeat_delay=0, push=False):
    """
    Animates an arbitrary array of 1D arrays using plt.plot.

    Parameters:
    - x: array of x values.
    - arrays: list of lists of 1D arrays to plot as a function of x. Each inner list corresponds to a different plot.
    - labels: list of labels for each plot.
    - interval: delay between frames in milliseconds (default is 200ms)
    - repeat: whether the animation should repeat when the sequence is completed (default is True)
    - repeat_delay: delay before repeating the animation in milliseconds (default is 0ms)
    """
    fig, ax = plt.subplots()

    # Initialize line objects for each plot
    lines = []
    for i in range(len(labels)):
        line, = ax.plot(x, arrays[i][0], label=labels[i])
        lines.append(line)

    ax.legend(loc='upper right')

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(x, arrays[i][frame])
        return lines

    ani = FuncAnimation(fig, update, frames=len(
        arrays[0]), interval=interval, repeat=repeat, repeat_delay=repeat_delay)
    if push:
        plt.show()
    else:
        return ani

# Example usage:
if __name__ == "__main__":
    # Create some example data: a list of 2D arrays
    data = [np.random.rand(10) for _ in range(30)]
    data2 = [np.random.rand(10) for _ in range(30)]

    # Call the function to animate the data
    animate_1d_lines(np.linspace(0, 10, 10), [
                     data, data2], labels=["hi", "bye"])

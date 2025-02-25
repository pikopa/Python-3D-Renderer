# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig, ax = plt.subplots()
scat = ax.scatter([0, 1], [0, 1])

def update(frame: int) -> tuple:
    points = np.random.rand(5, 2)  # Update points with some random noise
    scat.set_offsets(points)
    return (scat,)

# Define animation function
def animate(update):
    _ = animation.FuncAnimation(fig, update, frames=100, interval=1000, blit=True)

    plt.show()

if __name__ == "__main__":
    animate(update)
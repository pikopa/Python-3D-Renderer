import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Generate some data
points = np.random.rand(10, 2)  # Initial points

fig, ax = plt.subplots()
scat = ax.scatter(points[:, 0], points[:, 1])

def update(frame):
    global points
    points += np.random.randn(*points.shape) * 0.1  # Update points with some random noise
    scat.set_offsets(points)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=True)

plt.show()

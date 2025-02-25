# Import libraries
import time

import numpy as np
import matplotlib.pyplot as plt

from utils.project_gpu import project_using_gpu


class Screen:
    def __init__(self, size: tuple[int, int], origin: np.ndarray, delay: float = 0.0) -> None:
        self.width, self.height = size
        self.origin = origin
        self.delay = delay

        # Initializing a black screen
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Create a figure and axis
        self.figure, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.frame)

        # Set plt style
        plt.style.use("dark_background")
        # plt.axis("off")
        plt.ion()  # Enable interactive mode

        plt.show()

    def project(self, points: np.ndarray, depths: np.ndarray, colors: np.ndarray, frame: np.ndarray, z_buffer: np.ndarray, use_gpu: bool = False) -> np.ndarray:
        # Project points using GPU
        if use_gpu:
            return project_using_gpu(points, depths, colors, self.width, self.height, self.origin)

        # Normalize depth (avoids division by zero)
        depths = (depths - np.min(depths)) / (np.max(depths) - np.min(depths)) * 0.2 + 0.8

        # Perspective projection
        points[:, 0] /= depths
        points[:, 1] /= depths

        # Convert 3D points to screen coordinates
        points[:, 0] = points[:, 0] * self.width / 2 + self.origin[0]
        points[:, 1] = points[:, 1] * self.height / 2 + self.origin[1]

        # Convert to integer pixels
        points = points.astype(int)

        # Clamp coordinates to screen size
        points[:, 0] = np.clip(points[:, 0], 0, self.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.height - 1)

        # Render points with depth testing
        for i, (x, y) in enumerate(points[:, :2]):
            depth = depths[i]
            if depth < z_buffer[y, x]:  # Z-buffer test
                z_buffer[y, x] = depth
                frame[y, x, :] = colors[i]

        return frame

    def show(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.im.set_data(self.frame)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.im)

        self.figure.canvas.blit(self.ax.bbox)
        self.figure.canvas.flush_events()

        time.sleep(self.delay)


if __name__ == "__main__":
    # Example usage:
    screen = Screen((640, 480), 0.1)
    red = np.array([50, 50, 5])

    while True:
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        frame[red[0]:red[0] + red[2], red[1]:red[1] + red[2], :] = np.array([255, 0, 0])
        red[0] = (red[0] + 1) % 480
        red[1] = (red[1] + 1) % 640

        screen.show(frame)

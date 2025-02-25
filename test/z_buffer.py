import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

class Screen:
    def __init__(self, size: tuple[int, int], origin: np.ndarray, delay: float = 0.0) -> None:
        self.width, self.height = size
        self.origin = origin
        self.delay = delay

        # Initialize z-buffer with infinity values (far away from camera)
        self.z_buffer = np.full((self.height, self.width), np.inf)

        # Initializing a black screen
        self.frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Create a figure and axis
        self.figure, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.frame)

        # Set plt style
        plt.style.use("dark_background")
        plt.axis("off")
        plt.ion() # Enable interactive mode

        # Connect mouse events
        plt.connect("motion_notify_event", self.get_cursor)

        plt.show()

    @staticmethod
    def get_cursor(event: Any) -> np.ndarray:
        # Get mouse coordinates without clicking in matplotlib
        if event.xdata is None or event.ydata is None:
            return None
        return np.array([event.xdata, event.ydata])

    def project(self, points: np.ndarray, depths: np.ndarray, colors: np.ndarray) -> np.ndarray:
        # Project 2D points onto the screen
        points[:, 0] = points[:, 0] * self.width / 2 + self.origin[0]
        points[:, 1] = points[:, 1] * self.height / 2 + self.origin[1]

        # Convert points to integers
        points = points.astype(int)
        points = points[:, (0, 1)]

        # Initialize frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Create a mask to filter points inside the screen
        mask = (0 <= points[:, 0]) & (points[:, 0] < self.width) \
             & (0 <= points[:, 1]) & (points[:, 1] < self.height)

        # Apply the mask to get valid points and their corresponding depths and colors
        v_points = points[mask]
        v_colors = colors[mask]
        v_depths = depths[mask]

        # Perform depth testing using z-buffer
        for i, point in enumerate(v_points):
            x, y = point
            depth = v_depths[i]

            # Only update the pixel if the depth is closer than the current z-buffer value
            if depth < self.z_buffer[y, x]:
                self.z_buffer[y, x] = depth
                frame[y, x, :] = v_colors[i]

        return frame

    def show(self, frame: np.ndarray) -> None:
        # Update current frame
        self.frame = frame

        # Update figure
        self.im.set_data(self.frame)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.im)

        self.figure.canvas.blit(self.ax.bbox) # Update only the drawing area
        self.figure.canvas.flush_events()

        # Delay
        time.sleep(self.delay)


if __name__ == "__main__":
    # Example usage:
    screen = Screen((640, 480), 0.1)
    red = np.array([50, 50, 5])  # Starting position (x, y) and depth (z)

    while True:
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        depth = np.array([red[2]] * 4)  # Set depth for each corner
        color = np.array([255, 0, 0])  # Red color

        # Define the square (2D) and depth for each corner
        square_points = np.array([[red[0], red[1]],
                                  [red[0] + red[2], red[1]],
                                  [red[0], red[1] + red[2]],
                                  [red[0] + red[2], red[1] + red[2]]])

        frame = screen.project(square_points, depth, np.array([color] * 4))

        red[0] = (red[0] + 1) % 480
        red[1] = (red[1] + 1) % 640

        screen.show(frame)

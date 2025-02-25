# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plyfile import PlyData


# Set plt style
plt.style.use("dark_background")

# Read ply file
cube = PlyData.read("../data/chair.ply")
cube = np.array(cube.elements[0].data)
cube = np.array([[vertex[0], vertex[1], vertex[2]] for i, vertex in enumerate(cube) if i % 500 == 0])

# Shift cube to the center
cube -= np.mean(cube, axis=0)

# Define a screen
screen = {
    "origin": np.array([0, 0, 0]),
    "normal": np.array([0, 0, 1]),
}


def rotate(cube: np.ndarray, angle: float, direction: str) -> np.ndarray:
    """
    Rotate a cube in 3D space.

    Args:
        cube (np.ndarray): The cube to rotate.
        angle (float): The angle to rotate the cube by.

    Returns:
        np.ndarray: The rotated cube.

    """

    # Convert angle to radians
    radians = np.radians(angle)

    # Define rotation matri
    rotation_matrix = np.eye(3)

    rotate_x = np.array([
        [1, 0,               0               ],
        [0, np.cos(radians), -np.sin(radians)],
        [0, np.sin(radians), np.cos(radians) ],
    ])

    rotate_y = np.array([
        [np.cos(radians),  0, np.sin(radians)],  
        [0,                1,               0],
        [-np.sin(radians), 0, np.cos(radians)],
    ])

    rotate_z = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians),  0],
        [0,               0,                1],
    ])

    if "x" in direction:
        rotation_matrix = rotation_matrix @ rotate_x
    if "y" in direction:
        rotation_matrix = rotation_matrix @ rotate_y
    if "z" in direction:
        rotation_matrix = rotation_matrix @ rotate_z

    # Rotate the cube
    rotated_cube = np.dot(cube, rotation_matrix)

    return rotated_cube


def scene(cube: np.ndarray, screen: dict) -> None:
    """
    Plot a cube in 3D space.

    Args:
        cube (np.ndarray): The cube to plot.
        screen (dict): The screen to plot the cube on.

    Returns:
        None

    """

    # Create a figure and axis
    fig, ax = plt.subplots()
    fig.clf()

    # Set fixed axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")

    # Set default random color
    colors = np.tile([0, 1, 0], (cube.shape[0], 1))

    def update(frame: int) -> tuple:
        # Find the closest point to the screen for each vertex of the cube
        points = []
        distances = []

        for vertex in rotate(cube, frame, "xy"):
            # Find the point projected onto the screen
            point = vertex - np.dot(vertex - screen["origin"], screen["normal"]) * screen["normal"]

            # Calculate the distance to the screen
            distance = np.dot(vertex - np.array([0, 0, 0]), np.array([0, 1, 0]))

            points.append(point)
            distances.append(distance)

        # Plot the cube
        points = np.array(points)
        points = points[:, (0, 1)]

        # Calculate the scale factor to adjust the "distance" to the screen
        scale_factor = ((np.sin(np.radians(frame)) + 1) / 2 + 1)

        # Shift distance to positive
        distances = np.array(distances)
        distances = distances - np.min(distances)
        distances = distances / np.max(distances)
        distances *= 0.8
        distances += 0.2
        
        # Adjust colors
        colors_ = colors * distances[:, np.newaxis]

        # Plot the cube
        scat = ax.scatter(
            points[:, 0] * scale_factor,
            points[:, 1] * scale_factor,
            c=colors_,
            s=1,
        )

        return (scat,)

    # Create animation
    _ = animation.FuncAnimation(fig, update, frames=360, interval=10, blit=True)
    plt.show()


if __name__ == "__main__":
    scene(cube, screen)

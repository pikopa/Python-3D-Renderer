# Import libraries
import numpy as np

from cube import Cube
from screen import Screen


# Read ply file
point = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
], dtype=np.float32)

line = np.array([
    [0, 1],
    [0, 3],
    [0, 4],
    [1, 5],
    [2, 1],
    [2, 3],
    [3, 7],
    [4, 5],
    [4, 7],
    [6, 2],
    [6, 5],
    [6, 7],

    # [0, 6],
    # [1, 7],
    # [2, 4],
    # [3, 5],
])

# Define camera
camera = {
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


def render(cube: np.ndarray, colors: np.ndarray, camera: dict) -> None:
    """
    Plot a cube in 3D space.

    Args:
        cube (np.ndarray): The cube to plot.
        screen (dict): The screen to plot the cube on.

    Returns:
        None

    """

    # Define screen size
    screen = Screen(
        size=(500, 500),
        origin=(250, 250),
        delay=0.0,
    )

    for i in range(10000):
        # Rotate the cube
        points = rotate(cube=cube, angle=i, direction="xyz")

        # Perform the dot product and subtraction
        depths = np.dot(points - camera["origin"], camera["normal"])
        points = points - depths[:, np.newaxis] * camera["normal"]

        points *= ((np.sin(np.radians(i)) + 1) / 2 + 0)

        # Project the points onto the screen
        frame = screen.project(points=points, depths=depths, colors=colors)

        # Show the points
        screen.show(frame)


if __name__ == "__main__":
    cube = Cube()

    fine_grain = 0.002
    cube.add_points(point)
    cube.add_line(line, fine_grain, color=[0, 255, 0])
    # cube.add_plane([[0, 1, 2]], fine_grain, color=[255, 0, 0])
    # cube.add_plane([[2, 3, 0]], fine_grain, color=[255, 0, 0])
    # cube.add_plane([[4, 5, 6]], fine_grain, color=[0, 255, 0])
    # cube.add_plane([[6, 7, 4]], fine_grain, color=[0, 255, 0])
    # cube.add_plane([[0, 1, 5]], fine_grain, color=[0, 0, 255])
    # cube.add_plane([[5, 4, 0]], fine_grain, color=[0, 0, 255])
    # cube.add_plane([[1, 2, 6]], fine_grain, color=[255, 255, 0])
    # cube.add_plane([[6, 5, 1]], fine_grain, color=[255, 255, 0])
    # cube.add_plane([[2, 3, 6]], fine_grain, color=[255, 0, 255])
    # cube.add_plane([[6, 7, 3]], fine_grain, color=[255, 0, 255])
    # cube.add_plane([[0, 3, 7]], fine_grain, color=[0, 255, 255])
    # cube.add_plane([[7, 4, 0]], fine_grain, color=[0, 255, 255])

    points = cube.get_points()
    colors = cube.get_colors()

    render(points, colors, camera)

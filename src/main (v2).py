# Import libraries
import numpy as np

from screen import Screen
from utils.ply_reader import read_ply


# Read ply file
cube = np.array([
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

plane = np.array([
    [3, 0, 1],
    [5, 6, 7],
    # [6, 2, 0],
])

# Read ply file
# cube = read_ply("../data/chair.ply")

# Shift cube to the center
cube -= np.mean(cube, axis=0)

# Define camera
camera = {
    "origin": np.array([0, 0, 0]),
    "normal": np.array([0, 0, 1]),
}


def add_line(cube: np.ndarray, line: np.ndarray, density: float) -> np.ndarray:
    """
    Add interpolated points along the lines connecting cube vertices.

    Args:
        cube (np.ndarray): The cube vertices.
        line (np.ndarray): The pairs of indices defining edges.
        density (float): Distance between interpolated points.

    Returns:
        np.ndarray: New set of points including interpolated ones.

    """

    new_points = []
    
    for start_idx, end_idx in line:
        start_point = cube[start_idx]
        end_point = cube[end_idx]
        
        # Compute the vector along the line
        direction = end_point - start_point
        length = np.linalg.norm(direction)
        
        # Normalize direction
        direction = direction / length if length > 0 else direction
        
        # Number of points to insert
        num_points = int(length // density)
        
        # Generate interpolated points
        for i in range(1, num_points + 1):
            new_point = start_point + direction * (i * density)
            new_points.append(new_point)
    
    # Append new points to the original cube
    return np.vstack((cube, np.array(new_points, dtype=np.float32)))


def add_plane(cube: np.ndarray, plane: np.ndarray, density: float) -> np.ndarray:
    """
    Add interpolated points within the triangular planes defined by three vertices.

    Args:
        cube (np.ndarray): The cube vertices.
        plane (np.ndarray): The list of index triplets defining triangles.
        density (float): Distance between interpolated points.

    Returns:
        np.ndarray: New set of points including interpolated ones.
    """

    new_points = []

    for tri in plane:
        p0, p1, p2 = cube[tri[0]], cube[tri[1]], cube[tri[2]]

        # Compute triangle edges
        v1 = p1 - p0
        v2 = p2 - p0

        # Compute the triangle area
        area = np.linalg.norm(np.cross(v1, v2)) / 2

        # Estimate number of points to generate based on density and area
        num_points = max(1, int(area / (density ** 2)))

        # Generate random barycentric coordinates for uniform sampling
        for _ in range(num_points):
            r1, r2 = np.random.rand(), np.random.rand()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            new_point = p0 + r1 * v1 + r2 * v2
            new_points.append(new_point)

    return np.vstack((cube, np.array(new_points, dtype=np.float32)))

cube = add_line(cube, line, 0.005)
cube = add_plane(cube, plane, 0.005)

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


def main(cube: np.ndarray, camera: dict) -> None:
    """
    Plot a cube in 3D space.

    Args:
        cube (np.ndarray): The cube to plot.
        screen (dict): The screen to plot the cube on.

    Returns:
        None

    """

    # Define screen size
    screen = Screen(size=(200, 200), origin=(100, 100), delay=0.05)

    # Set default random color
    colors = np.tile([0, 255, 0], (cube.shape[0], 1))

    for i in range(10000):
        # Rotate the cube
        points = rotate(cube=cube, angle=i, direction="xy")

        # Perform the dot product and subtraction
        points = points - np.dot(points - camera["origin"], camera["normal"])[:, np.newaxis] * camera["normal"]

        # Scale cube size
        points = points * 1

        # Project the points onto the screen
        frame = screen.project(points, colors=colors)

        # Show the points
        screen.show(frame)


if __name__ == "__main__":
    main(cube, camera)

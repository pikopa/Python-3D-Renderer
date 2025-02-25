# Import libraries
import numpy as np

from cube import Cube
from screen import Screen

# Define camera
camera = {
    "origin": np.array([0, 0, 0]),
    "normal": np.array([0, 0, 1]),
}

def rotate(cube: Cube, angle: float, direction: str) -> np.ndarray:
    """
    Rotate a cube around its own center in 3D space.

    Args:
        cube (Cube): The cube instance to rotate.
        angle (float): The rotation angle in degrees.
        direction (str): Rotation axis ("x", "y", "z").

    Returns:
        np.ndarray: The rotated cube.
    """

    radians = np.radians(angle)

    # Define rotation matrices
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

    # Select rotation order
    rotation_matrix = np.eye(3)
    if "x" in direction:
        rotation_matrix = rotation_matrix @ rotate_x
    if "y" in direction:
        rotation_matrix = rotation_matrix @ rotate_y
    if "z" in direction:
        rotation_matrix = rotation_matrix @ rotate_z

    center = cube.get_origin()

    # Translate to origin, rotate, and move back
    translated_points = cube.get_points() - center
    rotated_points = translated_points @ rotation_matrix.T
    rotated_points += center

    return rotated_points


def render(cubes: list[Cube], camera: dict) -> None:
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
        size=(200, 200),
        origin=(100, 100),
        delay=0.0,
    )

    for iteration in range(10000):
        frame = np.zeros((screen.height, screen.width, 3), dtype=np.uint8)
        z_buffer = np.full((screen.height, screen.width), np.inf)

        use_gpu = True
        if use_gpu:
            points, colors, depths = [], [], []

            for i, cube in enumerate(cubes):
                points.append(rotate(cube=cube, angle=iteration, direction=["xz", "y", "xz"][i]) * 0.5)
                colors.append(cube.get_colors())
                depths.append(np.dot(points[-1] - camera["origin"], camera["normal"]))

            frame = screen.project(points, depths, colors, frame, z_buffer, use_gpu=use_gpu)
        else:
            for i, cube in enumerate(cubes):
                points = rotate(cube=cube, angle=iteration, direction=["xz", "y", "xz"][i]) * 0.5
                colors = cube.get_colors()
                depths = np.dot(points - camera["origin"], camera["normal"])

                # Render cube
                frame = screen.project(points, depths, colors, frame, z_buffer, use_gpu=use_gpu)

        # Display frame
        screen.show(frame)


if __name__ == "__main__":
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

    cube1 = Cube()
    fine_grain = 0.005
    cube1.add_points(point - np.mean(point))
    cube1.set_origin([-0.5, -0.5, 0])
    cube1.add_line(line, fine_grain, color=[0, 255, 0])
    cube1.add_plane([[0, 1, 2]], fine_grain, color=[255, 0, 0])
    cube1.add_plane([[2, 3, 0]], fine_grain, color=[255, 0, 0])
    cube1.add_plane([[4, 5, 6]], fine_grain, color=[0, 255, 0])
    cube1.add_plane([[6, 7, 4]], fine_grain, color=[0, 255, 0])
    cube1.add_plane([[0, 1, 5]], fine_grain, color=[0, 0, 255])
    cube1.add_plane([[5, 4, 0]], fine_grain, color=[0, 0, 255])
    # cube1.add_plane([[1, 2, 6]], fine_grain, color=[255, 255, 0])
    # cube1.add_plane([[6, 5, 1]], fine_grain, color=[255, 255, 0])
    # cube1.add_plane([[2, 3, 6]], fine_grain, color=[255, 0, 255])
    # cube1.add_plane([[6, 7, 3]], fine_grain, color=[255, 0, 255])
    # cube1.add_plane([[0, 3, 7]], fine_grain, color=[0, 255, 255])
    # cube1.add_plane([[7, 4, 0]], fine_grain, color=[0, 255, 255])

    cube2 = Cube()
    fine_grain = 0.005
    cube2.add_points(point)
    cube2.set_origin([0.5, 0.5, 0])
    cube2.add_line(line, fine_grain, color=[0, 255, 0])
    # cube2.add_plane([[0, 1, 2]], fine_grain, color=[255, 0, 0])
    # cube2.add_plane([[2, 3, 0]], fine_grain, color=[255, 0, 0])
    # cube2.add_plane([[4, 5, 6]], fine_grain, color=[0, 255, 0])
    # cube2.add_plane([[6, 7, 4]], fine_grain, color=[0, 255, 0])
    # cube2.add_plane([[0, 1, 5]], fine_grain, color=[0, 0, 255])
    # cube2.add_plane([[5, 4, 0]], fine_grain, color=[0, 0, 255])
    cube2.add_plane([[1, 2, 6]], fine_grain, color=[255, 255, 0])
    cube2.add_plane([[6, 5, 1]], fine_grain, color=[255, 255, 0])
    cube2.add_plane([[2, 3, 6]], fine_grain, color=[255, 0, 255])
    cube2.add_plane([[6, 7, 3]], fine_grain, color=[255, 0, 255])
    cube2.add_plane([[0, 3, 7]], fine_grain, color=[0, 255, 255])
    cube2.add_plane([[7, 4, 0]], fine_grain, color=[0, 255, 255])

    cube3 = Cube()
    fine_grain = 0.05
    cube3.set_origin([-1.5, 1.5, 0])
    cube3.add_points(np.array([
        [0, 0, 0],
        [3, 0, 0],
        [0, 0, 10],
        [3, 0, 10],
        [0, -1, 0],
        [3, -1, 0],
        [0, -1, 10],
        [3, -1, 10]
    ]), color=[255, 255, 255])
    cube3.add_plane([[0, 1, 2]], fine_grain, color=[255, 255, 255])
    cube3.add_plane([[1, 2, 3]], fine_grain, color=[255, 255, 255])
    cube3.add_line([[0, 4]], fine_grain, color=[255, 255, 255])
    cube3.add_line([[1, 5]], fine_grain, color=[255, 255, 255])
    cube3.add_line([[2, 6]], fine_grain, color=[255, 255, 255])
    cube3.add_line([[3, 7]], fine_grain, color=[255, 255, 255])


    from utils.ply_reader import read_ply
    cube4 = Cube()
    cube4.set_origin([0, 0, 0])
    cube4.add_points(read_ply("../data/chair.ply"))

    cubes = [cube4]

    render(cubes, camera)

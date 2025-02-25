# Import libraries
import numpy as np

from plyfile import PlyData


def read_ply(filename: str) -> np.ndarray:
    cube = PlyData.read(filename)
    cube = np.array(cube.elements[0].data)
    cube = np.array([[vertex[0], vertex[1], vertex[2]] for i, vertex in enumerate(cube) if i % 500 == 0])

    return cube
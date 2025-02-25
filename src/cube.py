# Import libraries
import numpy as np


class Cube:
    def __init__(self) -> None:
        self.origin = np.array([0, 0, 0])
        self.points = np.empty((0, 3), dtype=np.float32)
        self.colors = np.empty((0, 3), dtype=np.uint8)

    def set_origin(self, origin: np.ndarray) -> None:
        self.origin = origin

    def add_points(self, points: np.ndarray, color: np.ndarray = None) -> None:
        self.points = np.vstack((self.points, points))

        if color is None:
            color = np.array([0, 255, 0])

        self.colors = np.append(self.colors, np.tile(color, (points.shape[0], 1)), axis=0)

    def add_line(self, line: np.ndarray, density: float, color: np.ndarray = None) -> None:
        """
        Add interpolated points along the lines connecting cube vertices.

        Args:
            line (np.ndarray): The pairs of indices defining edges.
            density (float): Distance between interpolated points.

        Returns:
            np.ndarray: New set of points including interpolated ones.

        """

        new_points = []

        for start_idx, end_idx in line:
            start_point = self.points[start_idx]
            end_point = self.points[end_idx]

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
        self.points = np.vstack((self.points, new_points))

        if color is None:
            color = np.array([0, 255, 0])

        self.colors = np.append(self.colors, np.tile(color, (len(new_points), 1)), axis=0)

    def add_plane(self, plane: np.ndarray, density: float, color: np.ndarray = None) -> None:
        """
        Add interpolated points within the triangular planes defined by three vertices.

        Args:
            plane (np.ndarray): The list of index triplets defining triangles.
            density (float): Distance between interpolated points.

        Returns:
            np.ndarray: New set of points including interpolated ones.

        """

        new_points = []

        for tri in plane:
            p0, p1, p2 = self.points[tri[0]], self.points[tri[1]], self.points[tri[2]]

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

        # Append new points to the original cube
        self.points = np.append(self.points, new_points, axis=0)

        if color is None:
            color = np.array([0, 255, 0])

        self.colors = np.append(self.colors, np.tile(color, (len(new_points), 1)), axis=0)

    def get_origin(self) -> np.ndarray:
        return self.origin

    def get_points(self) -> np.ndarray:
        return self.points + self.origin

    def get_colors(self) -> np.ndarray:
        return self.colors

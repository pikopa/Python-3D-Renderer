# Import libraries
import numpy as np
from numba import cuda


# Define CUDA kernel
@cuda.jit
def project_points_gpu(
    points: np.ndarray, depths: np.ndarray, colors: np.ndarray,
    width: int, height: int, origin: np.ndarray,
    frame: np.ndarray, z_buffer: np.ndarray,
    min_depth: float, max_depth: float
) -> None:
    # 1D thread index
    i = cuda.grid(1)

    # Bounds check
    if i >= points.shape[0]:
        return

    # Normalize depth
    normalized_depth = (depths[i] - min_depth) / (max_depth - min_depth) * 0.2 + 0.8

    # Perspective projection
    px = points[i, 0] / normalized_depth
    py = points[i, 1] / normalized_depth

    # Convert to screen coordinates
    x = int(px * width / 2 + origin[0])
    y = int(py * height / 2 + origin[1])

    # Bounds check
    if 0 <= x < width and 0 <= y < height:
        depth = depths[i]

        # Use atomic min for thread-safe depth testing
        old_depth = cuda.atomic.min(z_buffer, (y, x), depth)

        # If this thread "won" the depth test, update the frame buffer
        if old_depth > depth:
            frame[y, x, 0] = colors[i, 0]
            frame[y, x, 1] = colors[i, 1]
            frame[y, x, 2] = colors[i, 2]


def project_using_gpu(
    points: list[np.ndarray], depths: list[np.ndarray], colors: list[np.ndarray],
    width: int, height: int, origin: np.ndarray
) -> np.ndarray:
    # Flatten all points, depths, and colors into single arrays
    points = np.vstack(points)
    depths = np.hstack(depths)
    colors = np.vstack(colors)

    # Compute min/max depth (to normalize depth values)
    min_depth = np.min(depths)
    max_depth = np.max(depths)

    # Prevent division by zero
    if max_depth - min_depth == 0:
        max_depth += 1e-5

    # Allocate GPU memory
    d_points = cuda.to_device(points)
    d_depths = cuda.to_device(depths)
    d_colors = cuda.to_device(colors)

    # Create frame buffer & Z-buffer on GPU
    d_frame = cuda.device_array((height, width, 3), dtype=np.uint8)
    d_z_buffer = cuda.device_array((height, width), dtype=np.float32)

    # Initialize buffers
    d_z_buffer[:] = np.inf  # Initialize Z-buffer to infinity
    d_frame[:] = 0  # Set frame to black

    # Define kernel launch configuration
    threads_per_block = 256
    blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block

    # Launch the CUDA kernel
    project_points_gpu[blocks_per_grid, threads_per_block](
        d_points, d_depths, d_colors,
        width, height, origin,
        d_frame, d_z_buffer,
        min_depth, max_depth,
    )

    # Copy result back to CPU
    frame = d_frame.copy_to_host()

    return frame

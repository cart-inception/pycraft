"""
Utility functions for the engine
Mathematical helpers and common operations
"""

import logging
import math
import time
from typing import Tuple, List, Optional
import numpy as np
from contextlib import contextmanager

from config import config

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing operations"""

    def __init__(self, name: str, warn_threshold_ms: float = 16.0):
        self.name = name
        self.warn_threshold_ms = warn_threshold_ms
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.start_time is not None:
            duration = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
            if duration > self.warn_threshold_ms:
                logger.warning(f"{self.name} took {duration:.2f}ms (threshold: {self.warn_threshold_ms}ms)")
            else:
                logger.debug(f"{self.name} took {duration:.2f}ms")


def create_perspective_matrix(fov: float, aspect_ratio: float, near: float, far: float) -> np.ndarray:
    """
    Create a perspective projection matrix

    Args:
        fov: Field of view in radians
        aspect_ratio: Width/height ratio
        near: Near clipping plane
        far: Far clipping plane

    Returns:
        4x4 perspective matrix
    """
    f = 1.0 / math.tan(fov / 2.0)
    matrix = np.zeros((4, 4), dtype=np.float32)

    matrix[0, 0] = f / aspect_ratio
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = -1.0
    matrix[3, 2] = (2.0 * far * near) / (near - far)

    return matrix


def create_look_at_matrix(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Create a look-at view matrix

    Args:
        eye: Camera position
        center: Point to look at
        up: Up vector

    Returns:
        4x4 view matrix
    """
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # Calculate forward, right, and up vectors
    forward = center - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    new_up = np.cross(right, forward)

    # Create view matrix
    matrix = np.eye(4, dtype=np.float32)

    matrix[0, 0] = right[0]
    matrix[1, 0] = right[1]
    matrix[2, 0] = right[2]

    matrix[0, 1] = new_up[0]
    matrix[1, 1] = new_up[1]
    matrix[2, 1] = new_up[2]

    matrix[0, 2] = -forward[0]
    matrix[1, 2] = -forward[1]
    matrix[2, 2] = -forward[2]

    matrix[3, 0] = -np.dot(right, eye)
    matrix[3, 1] = -np.dot(new_up, eye)
    matrix[3, 2] = np.dot(forward, eye)

    return matrix


def create_translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Create a translation matrix"""
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def create_scale_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Create a scale matrix"""
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = x
    matrix[1, 1] = y
    matrix[2, 2] = z
    return matrix


def create_rotation_matrix_x(angle: float) -> np.ndarray:
    """Create rotation matrix around X axis (angle in radians)"""
    matrix = np.eye(4, dtype=np.float32)
    c = math.cos(angle)
    s = math.sin(angle)
    matrix[1, 1] = c
    matrix[1, 2] = -s
    matrix[2, 1] = s
    matrix[2, 2] = c
    return matrix


def create_rotation_matrix_y(angle: float) -> np.ndarray:
    """Create rotation matrix around Y axis (angle in radians)"""
    matrix = np.eye(4, dtype=np.float32)
    c = math.cos(angle)
    s = math.sin(angle)
    matrix[0, 0] = c
    matrix[0, 2] = s
    matrix[2, 0] = -s
    matrix[2, 2] = c
    return matrix


def create_rotation_matrix_z(angle: float) -> np.ndarray:
    """Create rotation matrix around Z axis (angle in radians)"""
    matrix = np.eye(4, dtype=np.float32)
    c = math.cos(angle)
    s = math.sin(angle)
    matrix[0, 0] = c
    matrix[0, 1] = -s
    matrix[1, 0] = s
    matrix[1, 1] = c
    return matrix


def world_to_chunk(world_x: int, world_z: int) -> Tuple[int, int]:
    """
    Convert world coordinates to chunk coordinates

    Args:
        world_x: World X coordinate
        world_z: World Z coordinate

    Returns:
        (chunk_x, chunk_z) coordinates
    """
    chunk_x = world_x // config.CHUNK_SIZE
    if world_x < 0 and world_x % config.CHUNK_SIZE != 0:
        chunk_x -= 1

    chunk_z = world_z // config.CHUNK_SIZE
    if world_z < 0 and world_z % config.CHUNK_SIZE != 0:
        chunk_z -= 1

    return chunk_x, chunk_z


def chunk_to_world(chunk_x: int, chunk_z: int) -> Tuple[int, int]:
    """
    Convert chunk coordinates to world coordinates (chunk origin)

    Args:
        chunk_x: Chunk X coordinate
        chunk_z: Chunk Z coordinate

    Returns:
        (world_x, world_z) coordinates of chunk origin
    """
    return chunk_x * config.CHUNK_SIZE, chunk_z * config.CHUNK_SIZE


def block_in_chunk(local_x: int, local_y: int, local_z: int) -> bool:
    """
    Check if local coordinates are within chunk bounds

    Args:
        local_x: Local X coordinate
        local_y: Local Y coordinate
        local_z: Local Z coordinate

    Returns:
        True if coordinates are within chunk bounds
    """
    return (0 <= local_x < config.CHUNK_SIZE and
            0 <= local_y < config.CHUNK_HEIGHT and
            0 <= local_z < config.CHUNK_SIZE)


def distance_sq(pos1: List[float], pos2: List[float]) -> float:
    """Calculate squared distance between two positions"""
    return sum((a - b) ** 2 for a, b in zip(pos1, pos2))


def distance(pos1: List[float], pos2: List[float]) -> float:
    """Calculate distance between two positions"""
    return math.sqrt(distance_sq(pos1, pos2))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b"""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def rgba_to_int(r: int, g: int, b: int, a: int = 255) -> int:
    """Convert RGBA values to integer"""
    return (a << 24) | (r << 16) | (g << 8) | b


def int_to_rgba(value: int) -> Tuple[int, int, int, int]:
    """Convert integer to RGBA values"""
    a = (value >> 24) & 0xFF
    r = (value >> 16) & 0xFF
    g = (value >> 8) & 0xFF
    b = value & 0xFF
    return r, g, b, a


def get_memory_usage() -> float:
    """Get current memory usage in MB (rough estimate)"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # Fallback: just return 0 if psutil not available
        return 0.0


@contextmanager
def opengl_debug_group(name: str):
    """
    Context manager for grouping OpenGL operations (for debugging)
    Requires OpenGL debug output to be enabled
    """
    # This would be used with GL_KHR_debug extension
    # For now, it's just a placeholder for timing
    with Timer(f"OpenGL: {name}"):
        yield


def check_opengl_errors():
    """Check for OpenGL errors (debug builds only)"""
    if config.DEBUG_MODE:
        try:
            from OpenGL.GL import glGetError, GL_NO_ERROR
            error = glGetError()
            if error != GL_NO_ERROR:
                error_names = {
                    0x0500: "GL_INVALID_ENUM",
                    0x0501: "GL_INVALID_VALUE",
                    0x0502: "GL_INVALID_OPERATION",
                    0x0503: "GL_STACK_OVERFLOW",
                    0x0504: "GL_STACK_UNDERFLOW",
                    0x0505: "GL_OUT_OF_MEMORY"
                }
                error_name = error_names.get(error, f"Unknown error (0x{error:04X})")
                logger.error(f"OpenGL Error: {error_name}")
        except ImportError:
            pass


def format_bytes(bytes_count: int) -> str:
    """Format bytes in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} TB"


def validate_chunk_bounds(x: int, y: int, z: int) -> bool:
    """Validate that coordinates are within chunk bounds"""
    return block_in_chunk(x, y, z)


class PerformanceMonitor:
    """Simple performance monitoring for critical operations"""

    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self.samples = {}

    def add_sample(self, name: str, value: float):
        """Add a performance sample"""
        if name not in self.samples:
            self.samples[name] = []
        self.samples[name].append(value)

        # Keep only recent samples
        if len(self.samples[name]) > self.max_samples:
            self.samples[name] = self.samples[name][-self.max_samples:]

    def get_average(self, name: str) -> Optional[float]:
        """Get average value for a metric"""
        if name in self.samples and self.samples[name]:
            return sum(self.samples[name]) / len(self.samples[name])
        return None

    def get_stats(self, name: str) -> Optional[dict]:
        """Get statistics for a metric"""
        if name not in self.samples or not self.samples[name]:
            return None

        values = self.samples[name]
        return {
            'count': len(values),
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1]
        }


# Global performance monitor
performance_monitor = PerformanceMonitor()
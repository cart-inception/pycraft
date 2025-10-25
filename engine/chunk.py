"""
Chunk module - Voxel chunk data structure and management

This module implements the Chunk class which stores voxel data in a 3D NumPy array
and provides methods for block manipulation, coordinate conversion, and mesh
generation coordination.
"""

import numpy as np
import pickle
from typing import Tuple, Optional, Any
from dataclasses import dataclass

from config import Config
from engine.utils import world_to_chunk, chunk_to_world


@dataclass
class ChunkPosition:
    """Represents chunk coordinates in chunk space."""
    x: int
    z: int

    def __hash__(self):
        return hash((self.x, self.z))

    def __eq__(self, other):
        if not isinstance(other, ChunkPosition):
            return False
        return self.x == other.x and self.z == other.z

    def to_world_coords(self, local_x: int, y: int, local_z: int) -> Tuple[int, int, int]:
        """Convert local chunk coordinates to world coordinates."""
        world_x = self.x * Config.CHUNK_SIZE + local_x
        world_z = self.z * Config.CHUNK_SIZE + local_z
        return (world_x, y, world_z)

    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple for dictionary keys."""
        return (self.x, self.z)


class Chunk:
    """
    Represents a 16x256x16 chunk of voxel data.

    Uses NumPy array for efficient block storage and provides methods for
    block manipulation, coordinate conversion, and mesh generation coordination.
    """

    def __init__(self, position: ChunkPosition):
        """
        Initialize a new chunk.

        Args:
            position: ChunkPosition representing this chunk's location in the world
        """
        self.position = position
        self.size_x = Config.CHUNK_SIZE
        self.size_y = Config.CHUNK_HEIGHT
        self.size_z = Config.CHUNK_SIZE

        # 3D NumPy array for block IDs (uint8 for memory efficiency)
        # 0 = air, 1+ = block types
        self.blocks = np.zeros((self.size_x, self.size_y, self.size_z), dtype=np.uint8)

        # Metadata for mesh generation and rendering
        self.is_dirty = True  # Needs mesh regeneration
        self.mesh_vao = None  # OpenGL VAO handle for rendered mesh
        self.mesh_vbo = None  # OpenGL VBO handle for vertex data
        self.mesh_index_count = 0  # Number of indices to render

        # Performance tracking
        self.mesh_generation_time = 0.0
        self.last_access_time = 0.0

    def get_block(self, x: int, y: int, z: int) -> int:
        """
        Get block ID at local chunk coordinates.

        Args:
            x, y, z: Local coordinates within chunk (0 <= x < 16, 0 <= y < 256, 0 <= z < 16)

        Returns:
            Block ID (0 = air, 1+ = block types)
        """
        if not self._is_valid_local_coords(x, y, z):
            return 0  # Air for out-of-bounds

        return int(self.blocks[x, y, z])

    def set_block(self, x: int, y: int, z: int, block_id: int) -> bool:
        """
        Set block ID at local chunk coordinates.

        Args:
            x, y, z: Local coordinates within chunk
            block_id: Block ID to set (0 = air, 1+ = block types)

        Returns:
            True if block was set, False if coordinates were invalid
        """
        if not self._is_valid_local_coords(x, y, z):
            return False

        # Clamp block ID to valid range
        block_id = max(0, min(255, block_id))

        # Only mark dirty if block actually changed
        if self.blocks[x, y, z] != block_id:
            self.blocks[x, y, z] = block_id
            self.is_dirty = True

        return True

    def is_block_solid(self, x: int, y: int, z: int) -> bool:
        """
        Check if block at coordinates is solid (not air).

        Args:
            x, y, z: Local coordinates within chunk

        Returns:
            True if block is solid, False if air or out-of-bounds
        """
        if not self._is_valid_local_coords(x, y, z):
            return False

        return self.blocks[x, y, z] != 0

    def get_block_neighbor(self, x: int, y: int, z: int, direction: str) -> Tuple[int, int, int]:
        """
        Get neighboring block coordinates in chunk space.

        Args:
            x, y, z: Local coordinates
            direction: One of 'north', 'south', 'east', 'west', 'up', 'down'

        Returns:
            Tuple of (neighbor_x, neighbor_y, neighbor_z) coordinates
            May be outside chunk bounds
        """
        offsets = {
            'north': (0, 0, 1),
            'south': (0, 0, -1),
            'east': (1, 0, 0),
            'west': (-1, 0, 0),
            'up': (0, 1, 0),
            'down': (0, -1, 0)
        }

        if direction not in offsets:
            raise ValueError(f"Invalid direction: {direction}")

        dx, dy, dz = offsets[direction]
        return (x + dx, y + dy, z + dz)

    def is_on_chunk_boundary(self, x: int, y: int, z: int) -> bool:
        """
        Check if position is on chunk boundary (edge of chunk).

        Args:
            x, y, z: Local coordinates

        Returns:
            True if position is on any chunk face edge
        """
        return (x == 0 or x == self.size_x - 1 or
                y == 0 or y == self.size_y - 1 or
                z == 0 or z == self.size_z - 1)

    def get_boundary_directions(self, x: int, y: int, z: int) -> list:
        """
        Get list of directions that cross chunk boundaries from this position.

        Args:
            x, y, z: Local coordinates

        Returns:
            List of direction strings that cross chunk boundaries
        """
        directions = []

        if x == 0:
            directions.append('west')
        elif x == self.size_x - 1:
            directions.append('east')

        if z == 0:
            directions.append('south')
        elif z == self.size_z - 1:
            directions.append('north')

        if y == 0:
            directions.append('down')
        elif y == self.size_y - 1:
            directions.append('up')

        return directions

    def count_blocks(self, block_id: Optional[int] = None) -> int:
        """
        Count blocks in chunk.

        Args:
            block_id: Specific block ID to count, or None to count all non-air blocks

        Returns:
            Number of blocks matching criteria
        """
        if block_id is None:
            return np.count_nonzero(self.blocks)
        else:
            return np.count_nonzero(self.blocks == block_id)

    def fill_area(self, start: Tuple[int, int, int],
                  end: Tuple[int, int, int], block_id: int) -> None:
        """
        Fill a rectangular area with a block type.

        Args:
            start: (x, y, z) start coordinates (inclusive)
            end: (x, y, z) end coordinates (inclusive)
            block_id: Block ID to fill with
        """
        x1, y1, z1 = start
        x2, y2, z2 = end

        # Ensure coordinates are within bounds
        x1 = max(0, min(self.size_x - 1, x1))
        x2 = max(0, min(self.size_x - 1, x2))
        y1 = max(0, min(self.size_y - 1, y1))
        y2 = max(0, min(self.size_y - 1, y2))
        z1 = max(0, min(self.size_z - 1, z1))
        z2 = max(0, min(self.size_z - 1, z2))

        # Swap if start > end
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if z1 > z2:
            z1, z2 = z2, z1

        # Fill area
        self.blocks[x1:x2+1, y1:y2+1, z1:z2+1] = block_id
        self.is_dirty = True

    def clear(self) -> None:
        """Clear all blocks in chunk (set to air)."""
        self.blocks.fill(0)
        self.is_dirty = True

    def serialize(self) -> bytes:
        """
        Serialize chunk data to bytes for saving.

        Returns:
            Pickled bytes containing chunk data
        """
        data = {
            'position': self.position.to_tuple(),
            'blocks': self.blocks,
            'version': 1
        }
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, data: bytes) -> 'Chunk':
        """
        Deserialize chunk from bytes.

        Args:
            data: Pickled chunk data

        Returns:
            Chunk instance
        """
        chunk_data = pickle.loads(data)

        # Create chunk with stored position
        position = ChunkPosition(*chunk_data['position'])
        chunk = cls(position)

        # Restore block data
        chunk.blocks = chunk_data['blocks']

        # Mark as dirty since mesh needs to be regenerated
        chunk.is_dirty = True

        return chunk

    def _is_valid_local_coords(self, x: int, y: int, z: int) -> bool:
        """Check if coordinates are valid within this chunk."""
        return (0 <= x < self.size_x and
                0 <= y < self.size_y and
                0 <= z < self.size_z)

    def __repr__(self) -> str:
        """String representation of chunk."""
        block_count = self.count_blocks()
        return f"Chunk(position=({self.position.x}, {self.position.z}), blocks={block_count})"
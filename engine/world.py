"""
World module - Chunk management and coordinate system

This module implements the World class which manages chunks, handles coordinate
conversions between world and chunk space, and provides neighbor chunk queries
for mesh building across chunk boundaries.
"""

import logging
import time
from typing import Dict, Optional, Tuple, List, Set
from collections import deque

from config import Config
from engine.chunk import Chunk, ChunkPosition
from engine.mesh_builder import MeshBuilder
from blocks.block_types import is_block_solid

logger = logging.getLogger(__name__)


class World:
    """
    Manages the voxel world with chunk-based loading and coordinate systems.

    This class handles chunk storage, loading/unloading, coordinate conversions,
    and provides the interface for the mesh builder to query neighboring chunks.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the world.

        Args:
            seed: World seed for terrain generation (None = random)
        """
        self.seed = seed if seed is not None else int(time.time())
        self.chunks: Dict[Tuple[int, int], Chunk] = {}

        # Chunk management
        self.loaded_chunk_positions: Set[Tuple[int, int]] = set()
        self.chunk_load_queue: deque = deque()
        self.chunk_unload_queue: deque = deque()

        # Player position for chunk loading (default to origin)
        self.player_chunk_x = 0
        self.player_chunk_z = 0

        # Loading parameters
        self.load_radius = Config.RENDER_DISTANCE
        self.unload_radius = Config.RENDER_DISTANCE + 2  # Unload beyond render distance

        # Performance tracking
        self.total_chunks_generated = 0
        self.total_chunks_loaded = 0
        self.total_chunks_unloaded = 0
        self.chunk_generation_times: List[float] = []

        # Mesh rebuilding
        self.dirty_chunks: Set[Tuple[int, int]] = set()
        self.chunk_rebuild_queue: deque = deque()
        self.max_chunk_updates_per_frame = Config.MAX_CHUNK_UPDATES_PER_FRAME

        # Mesh builder for generating chunk meshes
        self.mesh_builder = MeshBuilder()

        logger.info(f"World initialized with seed: {self.seed}, "
                   f"load_radius: {self.load_radius}")

    def get_chunk_position(self, world_x: int, world_z: int) -> Tuple[int, int]:
        """
        Convert world coordinates to chunk coordinates.

        Args:
            world_x, world_z: World coordinates

        Returns:
            Tuple of (chunk_x, chunk_z) coordinates
        """
        chunk_x = world_x // Config.CHUNK_SIZE
        chunk_z = world_z // Config.CHUNK_SIZE

        # Handle negative coordinates correctly
        if world_x < 0 and world_x % Config.CHUNK_SIZE != 0:
            chunk_x -= 1
        if world_z < 0 and world_z % Config.CHUNK_SIZE != 0:
            chunk_z -= 1

        return (chunk_x, chunk_z)

    def get_local_chunk_coords(self, world_x: int, world_y: int, world_z: int) -> Tuple[int, int, int]:
        """
        Convert world coordinates to local chunk coordinates.

        Args:
            world_x, world_y, world_z: World coordinates

        Returns:
            Tuple of (local_x, local_y, local_z) coordinates within chunk
        """
        chunk_x, chunk_z = self.get_chunk_position(world_x, world_z)

        local_x = world_x - chunk_x * Config.CHUNK_SIZE
        local_y = world_y  # Y is not chunked
        local_z = world_z - chunk_z * Config.CHUNK_SIZE

        # Handle negative coordinates
        if local_x < 0:
            local_x += Config.CHUNK_SIZE
        if local_z < 0:
            local_z += Config.CHUNK_SIZE

        return (local_x, local_y, local_z)

    def get_chunk(self, chunk_x: int, chunk_z: int) -> Optional[Chunk]:
        """
        Get chunk at specified chunk coordinates.

        Args:
            chunk_x, chunk_z: Chunk coordinates

        Returns:
            Chunk instance if loaded, None otherwise
        """
        return self.chunks.get((chunk_x, chunk_z))

    def get_or_generate_chunk(self, chunk_x: int, chunk_z: int) -> Chunk:
        """
        Get chunk, generating it if it doesn't exist.

        Args:
            chunk_x, chunk_z: Chunk coordinates

        Returns:
            Chunk instance (always returns a valid chunk)
        """
        chunk_key = (chunk_x, chunk_z)

        if chunk_key in self.chunks:
            return self.chunks[chunk_key]

        # Generate new chunk
        start_time = time.perf_counter()
        chunk = self._generate_chunk(chunk_x, chunk_z)
        generation_time = time.perf_counter() - start_time

        # Store chunk
        self.chunks[chunk_key] = chunk
        self.loaded_chunk_positions.add(chunk_key)

        # Update performance stats
        self.total_chunks_generated += 1
        self.chunk_generation_times.append(generation_time)

        logger.debug(f"Generated chunk ({chunk_x}, {chunk_z}) in "
                    f"{generation_time*1000:.2f}ms")
        return chunk

    def is_block_solid_global(self, world_x: int, world_y: int, world_z: int) -> bool:
        """
        Check if block at world coordinates is solid.

        This method handles chunk boundaries and returns False for out-of-bounds
        positions, which is useful for mesh building face culling.

        Args:
            world_x, world_y, world_z: World coordinates

        Returns:
            True if block is solid, False if air or out-of-bounds
        """
        # Check world bounds
        if (world_y < 0 or world_y >= Config.CHUNK_HEIGHT or
            abs(world_x) > Config.CHUNK_SIZE * Config.RENDER_DISTANCE or
            abs(world_z) > Config.CHUNK_SIZE * Config.RENDER_DISTANCE):
            return False

        # Get chunk coordinates
        chunk_x, chunk_z = self.get_chunk_position(world_x, world_z)

        # Get or generate chunk if needed
        chunk = self.get_or_generate_chunk(chunk_x, chunk_z)

        # Convert to local coordinates
        local_x, local_y, local_z = self.get_local_chunk_coords(world_x, world_y, world_z)

        # Check if block is solid
        return chunk.is_block_solid(local_x, local_y, local_z)

    def get_block_global(self, world_x: int, world_y: int, world_z: int) -> int:
        """
        Get block ID at world coordinates.

        Args:
            world_x, world_y, world_z: World coordinates

        Returns:
            Block ID (0 = air, 1+ = block types)
        """
        # Check world bounds
        if (world_y < 0 or world_y >= Config.CHUNK_HEIGHT):
            return 0

        # Get chunk coordinates
        chunk_x, chunk_z = self.get_chunk_position(world_x, world_z)

        # Get chunk (generate if needed)
        chunk = self.get_or_generate_chunk(chunk_x, chunk_z)

        # Convert to local coordinates
        local_x, local_y, local_z = self.get_local_chunk_coords(world_x, world_y, world_z)

        # Get block
        return chunk.get_block(local_x, local_y, local_z)

    def set_block_global(self, world_x: int, world_y: int, world_z: int, block_id: int) -> bool:
        """
        Set block ID at world coordinates.

        Args:
            world_x, world_y, world_z: World coordinates
            block_id: Block ID to set

        Returns:
            True if block was set successfully
        """
        # Check world bounds
        if (world_y < 0 or world_y >= Config.CHUNK_HEIGHT):
            return False

        # Get chunk coordinates
        chunk_x, chunk_z = self.get_chunk_position(world_x, world_z)

        # Get or generate chunk
        chunk = self.get_or_generate_chunk(chunk_x, chunk_z)

        # Convert to local coordinates
        local_x, local_y, local_z = self.get_local_chunk_coords(world_x, world_y, world_z)

        # Set block
        success = chunk.set_block(local_x, local_y, local_z, block_id)

        # Mark neighboring chunks as dirty if on boundary
        if success and chunk.is_on_chunk_boundary(local_x, local_y, local_z):
            self._mark_neighbor_chunks_dirty(chunk_x, chunk_z, local_x, local_y, local_z)

        return success

    def get_neighbor_chunks(self, chunk_x: int, chunk_z: int) -> List[Optional[Chunk]]:
        """
        Get all 8 neighboring chunks around the specified chunk.

        Args:
            chunk_x, chunk_z: Center chunk coordinates

        Returns:
            List of 8 chunks in order: N, S, E, W, NE, NW, SE, SW
        """
        neighbors = []

        # Direct neighbors
        neighbors.append(self.get_chunk(chunk_x, chunk_z + 1))     # North
        neighbors.append(self.get_chunk(chunk_x, chunk_z - 1))     # South
        neighbors.append(self.get_chunk(chunk_x + 1, chunk_z))     # East
        neighbors.append(self.get_chunk(chunk_x - 1, chunk_z))     # West

        # Diagonal neighbors
        neighbors.append(self.get_chunk(chunk_x + 1, chunk_z + 1)) # NE
        neighbors.append(self.get_chunk(chunk_x - 1, chunk_z + 1)) # NW
        neighbors.append(self.get_chunk(chunk_x + 1, chunk_z - 1)) # SE
        neighbors.append(self.get_chunk(chunk_x - 1, chunk_z - 1)) # SW

        return neighbors

    def get_loaded_chunks_in_range(self, center_x: int, center_z: int,
                                  radius: int) -> List[Chunk]:
        """
        Get all loaded chunks within specified radius of center point.

        Args:
            center_x, center_z: Center chunk coordinates
            radius: Radius in chunks

        Returns:
            List of chunks within range
        """
        chunks_in_range = []

        for chunk_x in range(center_x - radius, center_x + radius + 1):
            for chunk_z in range(center_z - radius, center_z + radius + 1):
                chunk = self.get_chunk(chunk_x, chunk_z)
                if chunk is not None:
                    # Check if within circular radius
                    dist_sq = (chunk_x - center_x) ** 2 + (chunk_z - center_z) ** 2
                    if dist_sq <= radius ** 2:
                        chunks_in_range.append(chunk)

        return chunks_in_range

    def unload_chunk(self, chunk_x: int, chunk_z: int) -> bool:
        """
        Unload a chunk from memory.

        Args:
            chunk_x, chunk_z: Chunk coordinates

        Returns:
            True if chunk was unloaded, False if not found
        """
        chunk_key = (chunk_x, chunk_z)

        if chunk_key in self.chunks:
            chunk = self.chunks[chunk_key]

            # Clean up GPU resources if mesh exists
            if chunk.mesh_vao is not None:
                # Note: Actual GPU cleanup should be handled by renderer
                logger.debug(f"Chunk ({chunk_x}, {chunk_z}) has GPU resources that need cleanup")

            # Remove from world
            del self.chunks[chunk_key]
            self.loaded_chunk_positions.discard(chunk_key)

            logger.debug(f"Unloaded chunk ({chunk_x}, {chunk_z})")
            return True

        return False

    def update_player_position(self, world_x: float, world_z: float):
        """
        Update player position and trigger chunk loading/unloading if needed.

        Args:
            world_x, world_z: Player world coordinates
        """
        # Convert to chunk coordinates
        new_chunk_x, new_chunk_z = self.get_chunk_position(int(world_x), int(world_z))

        # Check if player moved to a different chunk
        if (new_chunk_x != self.player_chunk_x or
            new_chunk_z != self.player_chunk_z):

            self.player_chunk_x = new_chunk_x
            self.player_chunk_z = new_chunk_z

            # Update chunk loading
            self._update_chunk_loading()

    def _update_chunk_loading(self):
        """Update which chunks should be loaded based on player position."""
        # Clear existing queues
        self.chunk_load_queue.clear()
        self.chunk_unload_queue.clear()

        # Determine which chunks should be loaded
        chunks_to_load = self._get_chunks_in_radius(
            self.player_chunk_x, self.player_chunk_z, self.load_radius
        )

        # Queue chunks to load
        for chunk_x, chunk_z in chunks_to_load:
            if (chunk_x, chunk_z) not in self.chunks:
                self.chunk_load_queue.append((chunk_x, chunk_z))

        # Queue chunks to unload (those outside unload radius)
        chunks_to_unload = []
        for chunk_key in list(self.chunks.keys()):
            chunk_x, chunk_z = chunk_key
            distance_sq = ((chunk_x - self.player_chunk_x) ** 2 +
                          (chunk_z - self.player_chunk_z) ** 2)

            if distance_sq > self.unload_radius ** 2:
                chunks_to_unload.append((chunk_x, chunk_z))

        # Sort unload queue by distance (unload farthest first)
        chunks_to_unload.sort(key=lambda pos: (
            (pos[0] - self.player_chunk_x) ** 2 +
            (pos[1] - self.player_chunk_z) ** 2
        ), reverse=True)

        for chunk_x, chunk_z in chunks_to_unload:
            self.chunk_unload_queue.append((chunk_x, chunk_z))

        logger.debug(f"Chunk loading update: {len(self.chunk_load_queue)} to load, "
                    f"{len(self.chunk_unload_queue)} to unload")

    def _get_chunks_in_radius(self, center_x: int, center_z: int, radius: int) -> List[Tuple[int, int]]:
        """
        Get all chunk positions within specified radius using concentric circles.

        Args:
            center_x, center_z: Center chunk coordinates
            radius: Radius in chunks

        Returns:
            List of chunk coordinates sorted by distance from center
        """
        chunks = []

        # Load chunks in concentric circles (closest first)
        for dist in range(radius + 1):
            # Add chunks at current distance
            for dx in range(-dist, dist + 1):
                for dz in range(-dist, dist + 1):
                    # Only add chunks on the current circle boundary
                    if abs(dx) == dist or abs(dz) == dist:
                        chunk_x = center_x + dx
                        chunk_z = center_z + dz
                        chunks.append((chunk_x, chunk_z))

        # Sort by distance from center
        chunks.sort(key=lambda pos: (
            (pos[0] - center_x) ** 2 + (pos[1] - center_z) ** 2
        ))

        return chunks

    def process_chunk_loading(self, max_loads: int = 4, max_unloads: int = 8) -> Dict[str, int]:
        """
        Process chunk loading and unloading queues.

        Args:
            max_loads: Maximum chunks to load this frame
            max_unloads: Maximum chunks to unload this frame

        Returns:
            Dictionary with loading statistics
        """
        stats = {
            'chunks_loaded': 0,
            'chunks_unloaded': 0,
            'chunks_queued_for_load': len(self.chunk_load_queue),
            'chunks_queued_for_unload': len(self.chunk_unload_queue)
        }

        # Process unloads first (faster operation)
        unload_count = 0
        while self.chunk_unload_queue and unload_count < max_unloads:
            chunk_x, chunk_z = self.chunk_unload_queue.popleft()
            if self.unload_chunk(chunk_x, chunk_z):
                unload_count += 1
                self.total_chunks_unloaded += 1

        stats['chunks_unloaded'] = unload_count

        # Process loads
        load_count = 0
        while self.chunk_load_queue and load_count < max_loads:
            chunk_x, chunk_z = self.chunk_load_queue.popleft()
            if self.get_or_generate_chunk(chunk_x, chunk_z):
                load_count += 1
                self.total_chunks_loaded += 1

        stats['chunks_loaded'] = load_count
        stats['chunks_queued_for_load'] = len(self.chunk_load_queue)
        stats['chunks_queued_for_unload'] = len(self.chunk_unload_queue)

        if load_count > 0 or unload_count > 0:
            logger.debug(f"Processed chunk loading: {load_count} loaded, {unload_count} unloaded")

        return stats

    def force_load_chunks_in_radius(self, center_x: int, center_z: int, radius: int):
        """
        Force immediate loading of all chunks within radius.

        Args:
            center_x, center_z: Center chunk coordinates
            radius: Radius in chunks
        """
        chunks_to_load = self._get_chunks_in_radius(center_x, center_z, radius)

        logger.info(f"Force loading {len(chunks_to_load)} chunks in radius {radius}")

        for chunk_x, chunk_z in chunks_to_load:
            self.get_or_generate_chunk(chunk_x, chunk_z)

    def get_loading_status(self) -> Dict[str, any]:
        """
        Get current chunk loading status.

        Returns:
            Dictionary with loading status information
        """
        return {
            'player_chunk': (self.player_chunk_x, self.player_chunk_z),
            'loaded_chunks': len(self.chunks),
            'chunks_to_load': len(self.chunk_load_queue),
            'chunks_to_unload': len(self.chunk_unload_queue),
            'load_radius': self.load_radius,
            'unload_radius': self.unload_radius,
            'total_loaded': self.total_chunks_loaded,
            'total_unloaded': self.total_chunks_unloaded
        }

    def is_chunk_in_load_range(self, chunk_x: int, chunk_z: int) -> bool:
        """
        Check if chunk is within loading range of player.

        Args:
            chunk_x, chunk_z: Chunk coordinates

        Returns:
            True if chunk should be loaded
        """
        distance_sq = ((chunk_x - self.player_chunk_x) ** 2 +
                      (chunk_z - self.player_chunk_z) ** 2)
        return distance_sq <= self.load_radius ** 2

    def get_world_statistics(self) -> Dict[str, any]:
        """
        Get world statistics for debugging and monitoring.

        Returns:
            Dictionary with world statistics
        """
        total_blocks = sum(chunk.count_blocks() for chunk in self.chunks.values())

        avg_generation_time = 0
        if self.chunk_generation_times:
            avg_generation_time = sum(self.chunk_generation_times) / len(self.chunk_generation_times)

        return {
            'loaded_chunks': len(self.chunks),
            'total_blocks': total_blocks,
            'avg_blocks_per_chunk': total_blocks // len(self.chunks) if self.chunks else 0,
            'total_chunks_generated': self.total_chunks_generated,
            'total_chunks_loaded': self.total_chunks_loaded,
            'total_chunks_unloaded': self.total_chunks_unloaded,
            'avg_generation_time_ms': avg_generation_time * 1000,
            'world_seed': self.seed,
            'player_chunk': (self.player_chunk_x, self.player_chunk_z),
            'load_radius': self.load_radius
        }

    def _generate_chunk(self, chunk_x: int, chunk_z: int) -> Chunk:
        """
        Generate terrain for a chunk.

        This is a placeholder implementation that creates simple terrain.
        In a full implementation, this would use Perlin noise and more complex
        terrain generation algorithms.

        Args:
            chunk_x, chunk_z: Chunk coordinates

        Returns:
            Generated chunk
        """
        position = ChunkPosition(chunk_x, chunk_z)
        chunk = Chunk(position)

        # Simple heightmap terrain generation
        for x in range(Config.CHUNK_SIZE):
            for z in range(Config.CHUNK_SIZE):
                # Calculate world coordinates for noise
                world_x = chunk_x * Config.CHUNK_SIZE + x
                world_z = chunk_z * Config.CHUNK_SIZE + z

                # Generate height using simple sine waves (placeholder for Perlin noise)
                height = int(32 + 16 * (
                    0.5 * np.sin(world_x * 0.05) +
                    0.3 * np.sin(world_z * 0.08) +
                    0.2 * np.sin((world_x + world_z) * 0.03)
                ))

                # Clamp height
                height = max(1, min(Config.CHUNK_HEIGHT - 1, height))

                # Fill layers
                for y in range(height):
                    if y == height - 1:
                        # Top layer - grass
                        chunk.set_block(x, y, z, 1)
                    elif y >= height - 3:
                        # Dirt layer
                        chunk.set_block(x, y, z, 2)
                    else:
                        # Stone layer
                        chunk.set_block(x, y, z, 3)

        return chunk

    def _mark_neighbor_chunks_dirty(self, chunk_x: int, chunk_z: int,
                                    local_x: int, local_y: int, local_z: int):
        """
        Mark neighboring chunks as dirty if block change affects them.

        Args:
            chunk_x, chunk_z: Chunk coordinates
            local_x, local_y, local_z: Local coordinates within chunk
        """
        # Check which boundaries the block is on
        if local_x == 0 and chunk_x - 1 in self.chunks:
            self._mark_chunk_dirty(chunk_x - 1, chunk_z)
        elif local_x == Config.CHUNK_SIZE - 1 and chunk_x + 1 in self.chunks:
            self._mark_chunk_dirty(chunk_x + 1, chunk_z)

        if local_z == 0 and chunk_z - 1 in self.chunks:
            self._mark_chunk_dirty(chunk_x, chunk_z - 1)
        elif local_z == Config.CHUNK_SIZE - 1 and chunk_z + 1 in self.chunks:
            self._mark_chunk_dirty(chunk_x, chunk_z + 1)

    def _mark_chunk_dirty(self, chunk_x: int, chunk_z: int):
        """
        Mark a chunk as dirty and add it to the rebuild queue.

        Args:
            chunk_x, chunk_z: Chunk coordinates
        """
        chunk_key = (chunk_x, chunk_z)

        if chunk_key in self.chunks:
            chunk = self.chunks[chunk_key]
            if not chunk.is_dirty:
                chunk.is_dirty = True
                self.dirty_chunks.add(chunk_key)
                self.chunk_rebuild_queue.append(chunk_key)

    def mark_chunk_dirty(self, chunk_x: int, chunk_z: int):
        """
        Public method to mark a chunk as needing mesh rebuild.

        Args:
            chunk_x, chunk_z: Chunk coordinates
        """
        self._mark_chunk_dirty(chunk_x, chunk_z)

    def rebuild_chunk_mesh(self, chunk_x: int, chunk_z: int) -> Optional['MeshData']:
        """
        Rebuild mesh for a specific chunk.

        Args:
            chunk_x, chunk_z: Chunk coordinates

        Returns:
            MeshData if successful, None otherwise
        """
        chunk_key = (chunk_x, chunk_z)

        if chunk_key not in self.chunks:
            logger.warning(f"Attempted to rebuild mesh for non-existent chunk ({chunk_x}, {chunk_z})")
            return None

        chunk = self.chunks[chunk_key]

        if not chunk.is_dirty:
            return None  # No rebuild needed

        start_time = time.perf_counter()

        # Generate new mesh using neighbor query function
        mesh_data = self.mesh_builder.generate_chunk_mesh(
            chunk,
            world_neighbor_query=self.is_block_solid_global
        )

        # Update chunk mesh information
        chunk.mesh_generation_time = time.perf_counter() - start_time
        chunk.is_dirty = False
        chunk.mesh_index_count = len(mesh_data.indices)

        # Remove from dirty tracking
        self.dirty_chunks.discard(chunk_key)

        logger.debug(f"Rebuilt mesh for chunk ({chunk_x}, {chunk_z}): "
                    f"{mesh_data.vertex_count} vertices, "
                    f"{mesh_data.face_count} faces, "
                    f"{chunk.mesh_generation_time*1000:.2f}ms")

        return mesh_data

    def process_chunk_rebuilds(self, max_rebuilds: Optional[int] = None) -> Dict[str, int]:
        """
        Process chunk mesh rebuild queue, limiting updates per frame.

        Args:
            max_rebuilds: Maximum chunks to rebuild this frame (uses config default if None)

        Returns:
            Dictionary with rebuild statistics
        """
        if max_rebuilds is None:
            max_rebuilds = self.max_chunk_updates_per_frame

        stats = {
            'chunks_rebuilt': 0,
            'chunks_queued': len(self.chunk_rebuild_queue),
            'dirty_chunks': len(self.dirty_chunks)
        }

        rebuild_count = 0
        rebuilt_meshes = []

        # Process rebuild queue
        while self.chunk_rebuild_queue and rebuild_count < max_rebuilds:
            chunk_x, chunk_z = self.chunk_rebuild_queue.popleft()
            mesh_data = self.rebuild_chunk_mesh(chunk_x, chunk_z)

            if mesh_data:
                rebuilt_meshes.append(((chunk_x, chunk_z), mesh_data))
                rebuild_count += 1

        stats['chunks_rebuilt'] = rebuild_count
        stats['chunks_queued'] = len(self.chunk_rebuild_queue)
        stats['dirty_chunks'] = len(self.dirty_chunks)

        if rebuild_count > 0:
            logger.debug(f"Processed {rebuild_count} chunk mesh rebuilds")

        # Return mesh data for renderer to upload
        return stats, rebuilt_meshes

    def force_rebuild_all_chunks(self) -> List[Tuple[Tuple[int, int], 'MeshData']]:
        """
        Force rebuild all loaded chunk meshes.

        Returns:
            List of (chunk_coords, mesh_data) tuples for all rebuilt chunks
        """
        logger.info(f"Forcing rebuild of all {len(self.chunks)} loaded chunks")

        rebuilt_meshes = []

        # Mark all chunks as dirty
        for chunk_key in list(self.chunks.keys()):
            chunk_x, chunk_z = chunk_key
            self._mark_chunk_dirty(chunk_x, chunk_z)

        # Process all rebuilds
        while self.chunk_rebuild_queue:
            chunk_x, chunk_z = self.chunk_rebuild_queue.popleft()
            mesh_data = self.rebuild_chunk_mesh(chunk_x, chunk_z)

            if mesh_data:
                rebuilt_meshes.append(((chunk_x, chunk_z), mesh_data))

        logger.info(f"Rebuilt {len(rebuilt_meshes)} chunk meshes")
        return rebuilt_meshes

    def get_rebuild_status(self) -> Dict[str, any]:
        """
        Get current chunk rebuild status.

        Returns:
            Dictionary with rebuild status information
        """
        return {
            'dirty_chunks': len(self.dirty_chunks),
            'chunks_queued_for_rebuild': len(self.chunk_rebuild_queue),
            'max_updates_per_frame': self.max_chunk_updates_per_frame,
            'total_mesh_generations': self.mesh_builder.total_meshes_generated,
            'avg_mesh_generation_time_ms': (
                self.mesh_builder.total_generation_time / self.mesh_builder.total_meshes_generated * 1000
                if self.mesh_builder.total_meshes_generated > 0 else 0
            )
        }

    def set_max_chunk_updates_per_frame(self, max_updates: int):
        """
        Set maximum chunk mesh updates per frame.

        Args:
            max_updates: Maximum updates allowed per frame
        """
        self.max_chunk_updates_per_frame = max(1, max_updates)
        logger.info(f"Set max chunk updates per frame to {self.max_chunk_updates_per_frame}")

    def rebuild_chunk_and_neighbors(self, chunk_x: int, chunk_z: int) -> List[Tuple[Tuple[int, int], 'MeshData']]:
        """
        Rebuild a chunk and all its neighbors.

        Useful when block changes affect multiple chunks.

        Args:
            chunk_x, chunk_z: Center chunk coordinates

        Returns:
            List of (chunk_coords, mesh_data) tuples for rebuilt chunks
        """
        # Mark center chunk as dirty
        self._mark_chunk_dirty(chunk_x, chunk_z)

        # Mark all neighboring chunks as dirty
        neighbors = self.get_neighbor_chunks(chunk_x, chunk_z)
        neighbor_offsets = [
            (0, 1), (0, -1), (1, 0), (-1, 0),  # N, S, E, W
            (1, 1), (-1, 1), (1, -1), (-1, -1)  # Diagonals
        ]

        for i, neighbor_chunk in enumerate(neighbors):
            if neighbor_chunk is not None:
                dx, dz = neighbor_offsets[i]
                self._mark_chunk_dirty(chunk_x + dx, chunk_z + dz)

        # Process rebuilds for affected chunks
        rebuilt_meshes = []
        max_rebuilds = 10  # Allow more rebuilds for this operation

        rebuild_count = 0
        while self.chunk_rebuild_queue and rebuild_count < max_rebuilds:
            rebuild_chunk_x, rebuild_chunk_z = self.chunk_rebuild_queue.popleft()
            mesh_data = self.rebuild_chunk_mesh(rebuild_chunk_x, rebuild_chunk_z)

            if mesh_data:
                rebuilt_meshes.append(((rebuild_chunk_x, rebuild_chunk_z), mesh_data))
                rebuild_count += 1

        logger.debug(f"Rebuilt chunk and neighbors: {rebuild_count} chunks rebuilt")
        return rebuilt_meshes


# Import numpy for terrain generation
import numpy as np
"""
Mesh Builder module - Generates optimized meshes from voxel data

This module implements a greedy meshing algorithm to efficiently create meshes
from chunk data, reducing vertex count by merging adjacent faces of the same
block type.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from config import Config
from blocks.block_types import get_block_texture_uv, is_block_solid


@dataclass
class Vertex:
    """Represents a single vertex with position, UV coordinates, and lighting."""
    x: float
    y: float
    z: float
    u: float  # Texture U coordinate
    v: float  # Texture V coordinate
    light: float  # Lighting value (0.0 to 1.0)


@dataclass
class QuadFace:
    """Represents a quad face with 4 vertices."""
    vertices: List[Vertex]
    block_id: int
    normal: Tuple[int, int, int]  # Face normal vector


@dataclass
class MeshData:
    """Contains generated mesh data ready for upload to GPU."""
    vertices: List[float]  # Interleaved vertex data (x, y, z, u, v, light)
    indices: List[int]     # Index data for indexed drawing
    vertex_count: int      # Number of vertices
    face_count: int        # Number of faces (for debugging)
    generation_time: float # Time taken to generate mesh


class MeshBuilder:
    """
    Builds optimized meshes from chunk data using greedy meshing algorithm.

    This class takes chunk data and generates vertex and index buffers that can
    be uploaded to the GPU for rendering. The greedy meshing algorithm merges
    adjacent faces of the same block type to reduce draw calls.
    """

    def __init__(self):
        """Initialize mesh builder."""
        # Cache for frequently accessed data
        self._face_cache = {}
        self._uv_cache = {}

        # Performance metrics
        self.total_meshes_generated = 0
        self.total_generation_time = 0.0

    def generate_chunk_mesh(self, chunk, world_neighbor_query=None) -> MeshData:
        """
        Generate mesh data for a chunk.

        Args:
            chunk: Chunk instance to generate mesh for
            world_neighbor_query: Function to query block solidity in neighboring chunks
                                 Signature: (world_x, y, world_z) -> bool

        Returns:
            MeshData containing vertices and indices for rendering
        """
        start_time = time.perf_counter()

        # Clear previous mesh data
        vertices = []
        indices = []
        face_count = 0

        # Generate meshes for each axis-aligned plane
        # XZ planes (Y-axis faces - top/bottom)
        for y in range(Config.CHUNK_HEIGHT):
            self._greedy_mesh_xz_plane(chunk, y, vertices, indices, world_neighbor_query)

        # XY planes (Z-axis faces - front/back)
        for z in range(Config.CHUNK_SIZE):
            self._greedy_mesh_xy_plane(chunk, z, vertices, indices, world_neighbor_query)

        # YZ planes (X-axis faces - left/right)
        for x in range(Config.CHUNK_SIZE):
            self._greedy_mesh_yz_plane(chunk, x, vertices, indices, world_neighbor_query)

        generation_time = time.perf_counter() - start_time

        # Update performance metrics
        self.total_meshes_generated += 1
        self.total_generation_time += generation_time

        # Count faces (indices // 3 gives triangles, // 6 gives quads)
        face_count = len(indices) // 6

        return MeshData(
            vertices=vertices,
            indices=indices,
            vertex_count=len(vertices) // 6,  # 6 floats per vertex (x,y,z,u,v,light)
            face_count=face_count,
            generation_time=generation_time
        )

    def _greedy_mesh_xz_plane(self, chunk, y: int, vertices: List[float],
                              indices: List[int], world_neighbor_query=None):
        """
        Generate greedy mesh for XZ plane at height Y (top/bottom faces).

        Args:
            chunk: Chunk instance
            y: Y coordinate of the plane
            vertices: List to append vertex data to
            indices: List to append index data to
            world_neighbor_query: Function for neighbor chunk queries
        """
        # Create 2D array of block types for this plane
        plane = np.zeros((Config.CHUNK_SIZE, Config.CHUNK_SIZE), dtype=np.uint8)

        for x in range(Config.CHUNK_SIZE):
            for z in range(Config.CHUNK_SIZE):
                plane[x, z] = chunk.get_block(x, y, z)

        # Greedy meshing - find largest rectangles of same block type
        visited = np.zeros_like(plane, dtype=bool)

        for x in range(Config.CHUNK_SIZE):
            for z in range(Config.CHUNK_SIZE):
                if visited[x, z] or plane[x, z] == 0:
                    continue

                block_id = plane[x, z]

                # Check if top face should be rendered
                if self._should_render_face_top(chunk, x, y, z, world_neighbor_query):
                    # Find largest rectangle starting from (x, z)
                    width, height = self._find_largest_rectangle(plane, visited, x, z, block_id)

                    # Mark visited cells
                    for dx in range(width):
                        for dz in range(height):
                            visited[x + dx, z + dz] = True

                    # Generate quad for this rectangle
                    self._add_quad_top(vertices, indices, x, y, z, width, height, block_id)

                # Check if bottom face should be rendered
                elif self._should_render_face_bottom(chunk, x, y, z, world_neighbor_query):
                    # Find largest rectangle starting from (x, z)
                    width, height = self._find_largest_rectangle(plane, visited, x, z, block_id)

                    # Mark visited cells
                    for dx in range(width):
                        for dz in range(height):
                            visited[x + dx, z + dz] = True

                    # Generate quad for this rectangle
                    self._add_quad_bottom(vertices, indices, x, y, z, width, height, block_id)

    def _greedy_mesh_xy_plane(self, chunk, z: int, vertices: List[float],
                              indices: List[int], world_neighbor_query=None):
        """
        Generate greedy mesh for XY plane at Z coordinate (front/back faces).

        Args:
            chunk: Chunk instance
            z: Z coordinate of the plane
            vertices: List to append vertex data to
            indices: List to append index data to
            world_neighbor_query: Function for neighbor chunk queries
        """
        # Create 2D array of block types for this plane
        plane = np.zeros((Config.CHUNK_SIZE, Config.CHUNK_HEIGHT), dtype=np.uint8)

        for x in range(Config.CHUNK_SIZE):
            for y in range(Config.CHUNK_HEIGHT):
                plane[x, y] = chunk.get_block(x, y, z)

        # Greedy meshing
        visited = np.zeros_like(plane, dtype=bool)

        for x in range(Config.CHUNK_SIZE):
            for y in range(Config.CHUNK_HEIGHT):
                if visited[x, y] or plane[x, y] == 0:
                    continue

                block_id = plane[x, y]

                # Check if front face should be rendered
                if self._should_render_face_front(chunk, x, y, z, world_neighbor_query):
                    # Find largest rectangle
                    width, height = self._find_largest_rectangle(plane, visited, x, y, block_id)

                    # Mark visited cells
                    for dx in range(width):
                        for dy in range(height):
                            visited[x + dx, y + dy] = True

                    # Generate quad
                    self._add_quad_front(vertices, indices, x, y, z, width, height, block_id)

                # Check if back face should be rendered
                elif self._should_render_face_back(chunk, x, y, z, world_neighbor_query):
                    # Find largest rectangle
                    width, height = self._find_largest_rectangle(plane, visited, x, y, block_id)

                    # Mark visited cells
                    for dx in range(width):
                        for dy in range(height):
                            visited[x + dx, y + dy] = True

                    # Generate quad
                    self._add_quad_back(vertices, indices, x, y, z, width, height, block_id)

    def _greedy_mesh_yz_plane(self, chunk, x: int, vertices: List[float],
                              indices: List[int], world_neighbor_query=None):
        """
        Generate greedy mesh for YZ plane at X coordinate (left/right faces).

        Args:
            chunk: Chunk instance
            x: X coordinate of the plane
            vertices: List to append vertex data to
            indices: List to append index data to
            world_neighbor_query: Function for neighbor chunk queries
        """
        # Create 2D array of block types for this plane
        plane = np.zeros((Config.CHUNK_HEIGHT, Config.CHUNK_SIZE), dtype=np.uint8)

        for y in range(Config.CHUNK_HEIGHT):
            for z in range(Config.CHUNK_SIZE):
                plane[y, z] = chunk.get_block(x, y, z)

        # Greedy meshing
        visited = np.zeros_like(plane, dtype=bool)

        for y in range(Config.CHUNK_HEIGHT):
            for z in range(Config.CHUNK_SIZE):
                if visited[y, z] or plane[y, z] == 0:
                    continue

                block_id = plane[y, z]

                # Check if right face should be rendered
                if self._should_render_face_right(chunk, x, y, z, world_neighbor_query):
                    # Find largest rectangle
                    width, height = self._find_largest_rectangle(plane, visited, y, z, block_id)

                    # Mark visited cells
                    for dy in range(width):
                        for dz in range(height):
                            visited[y + dy, z + dz] = True

                    # Generate quad
                    self._add_quad_right(vertices, indices, x, y, z, width, height, block_id)

                # Check if left face should be rendered
                elif self._should_render_face_left(chunk, x, y, z, world_neighbor_query):
                    # Find largest rectangle
                    width, height = self._find_largest_rectangle(plane, visited, y, z, block_id)

                    # Mark visited cells
                    for dy in range(width):
                        for dz in range(height):
                            visited[y + dy, z + dz] = True

                    # Generate quad
                    self._add_quad_left(vertices, indices, x, y, z, width, height, block_id)

    def _find_largest_rectangle(self, plane: np.ndarray, visited: np.ndarray,
                                start_x: int, start_y: int, block_id: int) -> Tuple[int, int]:
        """
        Find the largest rectangle of the same block type starting from (start_x, start_y).

        Args:
            plane: 2D array of block types
            visited: Array marking visited cells
            start_x, start_y: Starting coordinates
            block_id: Block type to match

        Returns:
            Tuple of (width, height) of the largest rectangle
        """
        height, width = plane.shape

        # Start with 1x1 rectangle
        rect_width = 1
        rect_height = 1

        # Try to expand width first (with bounds checking)
        while (start_x + rect_width < width and
               start_y < height and
               start_x + rect_width >= 0 and start_y >= 0 and
               not visited[start_x + rect_width, start_y] and
               plane[start_x + rect_width, start_y] == block_id):
            rect_width += 1

        # Try to expand height, checking all columns in the current width (with bounds checking)
        while (start_y + rect_height < height and
               all(start_x + x < width and
                   not visited[start_x + x, start_y + rect_height] and
                   plane[start_x + x, start_y + rect_height] == block_id
                   for x in range(rect_width))):
            rect_height += 1

        # Try to optimize by finding a better rectangle (with proper bounds checking)
        best_width = rect_width
        best_height = rect_height

        for test_height in range(1, rect_height + 1):
            for test_width in range(1, rect_width + 1):
                if test_width * test_height > best_width * best_height:
                    # Check if this rectangle is valid and within bounds
                    if (start_x + test_width <= width and
                        start_y + test_height <= height):
                        if all(all(start_x + x < width and
                                  start_y + y < height and
                                  not visited[start_x + x, start_y + y] and
                                  plane[start_x + x, start_y + y] == block_id
                                  for x in range(test_width))
                               for y in range(test_height)):
                            best_width = test_width
                            best_height = test_height

        return (best_width, best_height)

    def _should_render_face_top(self, chunk, x: int, y: int, z: int, world_neighbor_query) -> bool:
        """Check if top face should be rendered."""
        if y >= Config.CHUNK_HEIGHT - 1:
            # At chunk top, check world above
            if world_neighbor_query:
                world_x, world_y, world_z = chunk.position.to_world_coords(x, y + 1, z)
                return not is_block_solid(world_neighbor_query(world_x, world_y, world_z))
            return True  # Assume air above if no neighbor query

        # Check block above in same chunk
        return not chunk.is_block_solid(x, y + 1, z)

    def _should_render_face_bottom(self, chunk, x: int, y: int, z: int, world_neighbor_query) -> bool:
        """Check if bottom face should be rendered."""
        if y <= 0:
            # At chunk bottom, check world below
            if world_neighbor_query:
                world_x, world_y, world_z = chunk.position.to_world_coords(x, y - 1, z)
                return not is_block_solid(world_neighbor_query(world_x, world_y, world_z))
            return True  # Assume air below if no neighbor query

        # Check block below in same chunk
        return not chunk.is_block_solid(x, y - 1, z)

    def _should_render_face_front(self, chunk, x: int, y: int, z: int, world_neighbor_query) -> bool:
        """Check if front face (+Z) should be rendered."""
        if z >= Config.CHUNK_SIZE - 1:
            # At chunk front edge, check neighboring chunk
            if world_neighbor_query:
                world_x, world_y, world_z = chunk.position.to_world_coords(x, y, z + 1)
                return not is_block_solid(world_neighbor_query(world_x, world_y, world_z))
            return True

        # Check block in front in same chunk
        return not chunk.is_block_solid(x, y, z + 1)

    def _should_render_face_back(self, chunk, x: int, y: int, z: int, world_neighbor_query) -> bool:
        """Check if back face (-Z) should be rendered."""
        if z <= 0:
            # At chunk back edge, check neighboring chunk
            if world_neighbor_query:
                world_x, world_y, world_z = chunk.position.to_world_coords(x, y, z - 1)
                return not is_block_solid(world_neighbor_query(world_x, world_y, world_z))
            return True

        # Check block behind in same chunk
        return not chunk.is_block_solid(x, y, z - 1)

    def _should_render_face_right(self, chunk, x: int, y: int, z: int, world_neighbor_query) -> bool:
        """Check if right face (+X) should be rendered."""
        if x >= Config.CHUNK_SIZE - 1:
            # At chunk right edge, check neighboring chunk
            if world_neighbor_query:
                world_x, world_y, world_z = chunk.position.to_world_coords(x + 1, y, z)
                return not is_block_solid(world_neighbor_query(world_x, world_y, world_z))
            return True

        # Check block to the right in same chunk
        return not chunk.is_block_solid(x + 1, y, z)

    def _should_render_face_left(self, chunk, x: int, y: int, z: int, world_neighbor_query) -> bool:
        """Check if left face (-X) should be rendered."""
        if x <= 0:
            # At chunk left edge, check neighboring chunk
            if world_neighbor_query:
                world_x, world_y, world_z = chunk.position.to_world_coords(x - 1, y, z)
                return not is_block_solid(world_neighbor_query(world_x, world_y, world_z))
            return True

        # Check block to the left in same chunk
        return not chunk.is_block_solid(x - 1, y, z)

    def _add_quad_top(self, vertices: List[float], indices: List[int],
                      x: int, y: int, z: int, width: int, depth: int, block_id: int):
        """Add top-facing quad to mesh data."""
        # Get texture UV coordinates
        u, v, tex_width, tex_height = get_block_texture_uv(block_id, 'top')

        # Calculate UV offsets for quad
        u_offset = 0.0
        v_offset = 0.0
        u_scale = tex_width * width
        v_scale = tex_height * depth

        # Define quad vertices (top face, looking down)
        quad_vertices = [
            # Vertex format: x, y, z, u, v, light
            x, y + 1, z, u_offset, v_offset, 1.0,           # Bottom-left
            x + width, y + 1, z, u_scale, v_offset, 1.0,     # Bottom-right
            x + width, y + 1, z + depth, u_scale, v_scale, 1.0,  # Top-right
            x, y + 1, z + depth, u_offset, v_scale, 1.0,    # Top-left
        ]

        # Add vertices
        base_index = len(vertices) // 6
        vertices.extend(quad_vertices)

        # Add indices for two triangles
        indices.extend([
            base_index, base_index + 1, base_index + 2,
            base_index, base_index + 2, base_index + 3
        ])

    def _add_quad_bottom(self, vertices: List[float], indices: List[int],
                         x: int, y: int, z: int, width: int, depth: int, block_id: int):
        """Add bottom-facing quad to mesh data."""
        # Get texture UV coordinates
        u, v, tex_width, tex_height = get_block_texture_uv(block_id, 'bottom')

        # Calculate UV offsets
        u_offset = 0.0
        v_offset = 0.0
        u_scale = tex_width * width
        v_scale = tex_height * depth

        # Define quad vertices (bottom face, looking up)
        quad_vertices = [
            x, y, z + depth, u_offset, v_offset, 0.7,      # Top-left (but flipped)
            x + width, y, z + depth, u_scale, v_offset, 0.7,  # Top-right
            x + width, y, z, u_scale, v_scale, 0.7,        # Bottom-right
            x, y, z, u_offset, v_scale, 0.7,              # Bottom-left
        ]

        # Add vertices
        base_index = len(vertices) // 6
        vertices.extend(quad_vertices)

        # Add indices
        indices.extend([
            base_index, base_index + 1, base_index + 2,
            base_index, base_index + 2, base_index + 3
        ])

    def _add_quad_front(self, vertices: List[float], indices: List[int],
                       x: int, y: int, z: int, width: int, height: int, block_id: int):
        """Add front-facing quad (+Z) to mesh data."""
        # Get texture UV coordinates
        u, v, tex_width, tex_height = get_block_texture_uv(block_id, 'sides')

        # Calculate UV offsets
        u_offset = 0.0
        v_offset = 0.0
        u_scale = tex_width * width
        v_scale = tex_height * height

        # Define quad vertices (front face)
        quad_vertices = [
            x, y, z + 1, u_offset, v_offset, 0.9,           # Bottom-left
            x + width, y, z + 1, u_scale, v_offset, 0.9,     # Bottom-right
            x + width, y + height, z + 1, u_scale, v_scale, 0.9,  # Top-right
            x, y + height, z + 1, u_offset, v_scale, 0.9,    # Top-left
        ]

        # Add vertices
        base_index = len(vertices) // 6
        vertices.extend(quad_vertices)

        # Add indices
        indices.extend([
            base_index, base_index + 1, base_index + 2,
            base_index, base_index + 2, base_index + 3
        ])

    def _add_quad_back(self, vertices: List[float], indices: List[int],
                      x: int, y: int, z: int, width: int, height: int, block_id: int):
        """Add back-facing quad (-Z) to mesh data."""
        # Get texture UV coordinates
        u, v, tex_width, tex_height = get_block_texture_uv(block_id, 'sides')

        # Calculate UV offsets
        u_offset = 0.0
        v_offset = 0.0
        u_scale = tex_width * width
        v_scale = tex_height * height

        # Define quad vertices (back face)
        quad_vertices = [
            x + width, y, z, u_offset, v_offset, 0.8,       # Bottom-left
            x, y, z, u_scale, v_offset, 0.8,               # Bottom-right
            x, y + height, z, u_scale, v_scale, 0.8,       # Top-right
            x + width, y + height, z, u_offset, v_scale, 0.8,  # Top-left
        ]

        # Add vertices
        base_index = len(vertices) // 6
        vertices.extend(quad_vertices)

        # Add indices
        indices.extend([
            base_index, base_index + 1, base_index + 2,
            base_index, base_index + 2, base_index + 3
        ])

    def _add_quad_right(self, vertices: List[float], indices: List[int],
                       x: int, y: int, z: int, depth: int, height: int, block_id: int):
        """Add right-facing quad (+X) to mesh data."""
        # Get texture UV coordinates
        u, v, tex_width, tex_height = get_block_texture_uv(block_id, 'sides')

        # Calculate UV offsets
        u_offset = 0.0
        v_offset = 0.0
        u_scale = tex_width * depth
        v_scale = tex_height * height

        # Define quad vertices (right face)
        quad_vertices = [
            x + 1, y, z + depth, u_offset, v_offset, 0.85,  # Bottom-left
            x + 1, y, z, u_scale, v_offset, 0.85,           # Bottom-right
            x + 1, y + height, z, u_scale, v_scale, 0.85,   # Top-right
            x + 1, y + height, z + depth, u_offset, v_scale, 0.85,  # Top-left
        ]

        # Add vertices
        base_index = len(vertices) // 6
        vertices.extend(quad_vertices)

        # Add indices
        indices.extend([
            base_index, base_index + 1, base_index + 2,
            base_index, base_index + 2, base_index + 3
        ])

    def _add_quad_left(self, vertices: List[float], indices: List[int],
                      x: int, y: int, z: int, depth: int, height: int, block_id: int):
        """Add left-facing quad (-X) to mesh data."""
        # Get texture UV coordinates
        u, v, tex_width, tex_height = get_block_texture_uv(block_id, 'sides')

        # Calculate UV offsets
        u_offset = 0.0
        v_offset = 0.0
        u_scale = tex_width * depth
        v_scale = tex_height * height

        # Define quad vertices (left face)
        quad_vertices = [
            x, y, z, u_offset, v_offset, 0.75,              # Bottom-left
            x, y, z + depth, u_scale, v_offset, 0.75,      # Bottom-right
            x, y + height, z + depth, u_scale, v_scale, 0.75,  # Top-right
            x, y + height, z, u_offset, v_scale, 0.75,     # Top-left
        ]

        # Add vertices
        base_index = len(vertices) // 6
        vertices.extend(quad_vertices)

        # Add indices
        indices.extend([
            base_index, base_index + 1, base_index + 2,
            base_index, base_index + 2, base_index + 3
        ])

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.total_meshes_generated == 0:
            return {
                'total_meshes': 0,
                'avg_generation_time': 0.0,
                'total_generation_time': 0.0
            }

        return {
            'total_meshes': self.total_meshes_generated,
            'avg_generation_time': self.total_generation_time / self.total_meshes_generated,
            'total_generation_time': self.total_generation_time
        }
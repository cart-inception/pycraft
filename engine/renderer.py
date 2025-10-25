"""
Basic 3D rendering system for Pycraft
Handles VAO/VBO management, mesh creation, and rendering operations
"""

import logging
import ctypes
import numpy as np
from typing import List, Tuple, Optional, Dict

from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT,
    GL_UNSIGNED_INT, GL_TRIANGLES, GL_FALSE, glGenVertexArrays, glGenBuffers,
    glBindVertexArray, glBindBuffer, glBufferData, glEnableVertexAttribArray,
    glVertexAttribPointer, glDeleteVertexArrays, glDeleteBuffers, glDrawElements,
    GL_TEXTURE_2D, glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_NEAREST, GL_REPEAT,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_RGBA, GL_UNSIGNED_BYTE, glActiveTexture,
    GL_TEXTURE0, glDeleteTextures
)

from config import config
from engine.utils import Timer, check_opengl_errors
from engine.mesh_builder import MeshData

logger = logging.getLogger(__name__)


class Mesh:
    """
    Simple mesh container for vertex data and indices
    """

    def __init__(self, vertices: np.ndarray, indices: np.ndarray,
                 tex_coords: np.ndarray = None, colors: np.ndarray = None):
        """
        Initialize mesh with vertex data

        Args:
            vertices: Nx3 array of vertex positions
            indices: Array of indices for indexed drawing
            tex_coords: Nx2 array of texture coordinates (optional)
            colors: Nx3 array of vertex colors (optional)
        """
        self.vertices = vertices.astype(np.float32)
        self.indices = indices.astype(np.uint32)
        self.tex_coords = tex_coords.astype(np.float32) if tex_coords is not None else None
        self.colors = colors.astype(np.float32) if colors is not None else None

        self.vao: Optional[int] = None
        self.vbo: Optional[int] = None
        self.ebo: Optional[int] = None
        self.tex_coord_vbo: Optional[int] = None
        self.color_vbo: Optional[int] = None

        self._upload_to_gpu()


class ChunkMesh:
    """
    Optimized mesh container for chunk rendering with interleaved vertex data
    """

    def __init__(self, mesh_data: MeshData):
        """
        Initialize chunk mesh from mesh data

        Args:
            mesh_data: MeshData containing vertices and indices from mesh builder
        """
        self.vertex_count = mesh_data.vertex_count
        self.index_count = len(mesh_data.indices)
        self.face_count = mesh_data.face_count
        self.generation_time = mesh_data.generation_time

        # Convert vertex data to numpy array
        vertices = np.array(mesh_data.vertices, dtype=np.float32)
        indices = np.array(mesh_data.indices, dtype=np.uint32)

        self.vao: Optional[int] = None
        self.vbo: Optional[int] = None
        self.ebo: Optional[int] = None

        self._upload_to_gpu(vertices, indices)

    def _upload_to_gpu(self, vertices: np.ndarray, indices: np.ndarray):
        """Upload chunk mesh data to GPU with interleaved vertex format"""
        # Generate and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Generate and bind VBO for interleaved vertex data
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Set vertex attribute pointers for interleaved format
        # Layout: position (3 floats), uv (2 floats), light (1 float)
        vertex_size = 6 * 4  # 6 floats * 4 bytes each = 24 bytes

        # Position attribute (location 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, None)

        # UV coordinate attribute (location 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertex_size,
                            ctypes.c_void_p(3 * 4))  # Offset by position

        # Light attribute (location 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, vertex_size,
                            ctypes.c_void_p(5 * 4))  # Offset by position + UV

        # Generate and bind EBO for indices
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Unbind VAO
        glBindVertexArray(0)

        check_opengl_errors()
        logger.debug(f"Chunk mesh uploaded to GPU: {self.vertex_count} vertices, "
                    f"{self.index_count} indices, {self.face_count} faces")

    def draw(self):
        """Draw the chunk mesh"""
        if self.vao is None:
            logger.warning("Attempting to draw chunk mesh that hasn't been uploaded to GPU")
            return

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def cleanup(self):
        """Clean up GPU resources"""
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        if self.vbo is not None:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = None
        if self.ebo is not None:
            glDeleteBuffers(1, [self.ebo])
            self.ebo = None

    def _upload_to_gpu(self):
        """Upload mesh data to GPU"""
        # Generate and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Generate and bind VBO for vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Set vertex attribute pointer for position (location 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)

        # Generate and bind EBO for indices
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # Upload texture coordinates if provided
        if self.tex_coords is not None:
            self.tex_coord_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.tex_coord_vbo)
            glBufferData(GL_ARRAY_BUFFER, self.tex_coords.nbytes, self.tex_coords, GL_STATIC_DRAW)

            # Set vertex attribute pointer for texture coordinates (location 1)
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * 4, None)

        # Upload colors if provided
        if self.colors is not None:
            self.color_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.color_vbo)
            glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_STATIC_DRAW)

            # Set vertex attribute pointer for colors (location 2)
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * 4, None)

        # Unbind VAO
        glBindVertexArray(0)

        check_opengl_errors()
        logger.debug(f"Mesh uploaded to GPU: {len(self.vertices)} vertices, {len(self.indices)} indices")

    def draw(self):
        """Draw the mesh"""
        if self.vao is None:
            logger.warning("Attempting to draw mesh that hasn't been uploaded to GPU")
            return

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def cleanup(self):
        """Clean up GPU resources"""
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        if self.vbo is not None:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = None
        if self.ebo is not None:
            glDeleteBuffers(1, [self.ebo])
            self.ebo = None
        if self.tex_coord_vbo is not None:
            glDeleteBuffers(1, [self.tex_coord_vbo])
            self.tex_coord_vbo = None
        if self.color_vbo is not None:
            glDeleteBuffers(1, [self.color_vbo])
            self.color_vbo = None


class Texture:
    """
    Simple 2D texture wrapper
    """

    def __init__(self, width: int, height: int, data: np.ndarray = None):
        """
        Initialize texture

        Args:
            width: Texture width
            height: Texture height
            data: RGBA pixel data (numpy array, optional)
        """
        self.width = width
        self.height = height
        self.texture_id: Optional[int] = None

        if data is None:
            # Generate a simple colored texture as default
            data = self._generate_default_texture()

        self._upload_to_gpu(data)

    def _generate_default_texture(self) -> np.ndarray:
        """Generate a simple default texture"""
        data = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        # Create a simple checkerboard pattern with colors
        for y in range(self.height):
            for x in range(self.width):
                if (x // 8 + y // 8) % 2 == 0:
                    data[y, x] = [255, 100, 100, 255]  # Red
                else:
                    data[y, x] = [100, 100, 255, 255]  # Blue

        return data

    def _upload_to_gpu(self, data: np.ndarray):
        """Upload texture data to GPU"""
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        # Upload texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height,
                    0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        glBindTexture(GL_TEXTURE_2D, 0)
        check_opengl_errors()
        logger.debug(f"Texture uploaded to GPU: {self.width}x{self.height}")

    def bind(self, texture_unit: int = 0):
        """Bind texture to specified texture unit"""
        if self.texture_id is not None:
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)

    def cleanup(self):
        """Clean up GPU resources"""
        if self.texture_id is not None:
            glDeleteTextures(1, [self.texture_id])
            self.texture_id = None


class Renderer:
    """
    Basic renderer for 3D meshes with chunk mesh support
    """

    def __init__(self):
        """Initialize renderer"""
        self.meshes: Dict[str, Mesh] = {}
        self.textures: Dict[str, Texture] = {}
        self.chunk_meshes: Dict[str, ChunkMesh] = {}

        # VBO pooling for memory efficiency
        self._vbo_pool: List[int] = []
        self._vao_pool: List[int] = []
        self._ebo_pool: List[int] = []

        logger.info("Renderer initialized")

    def create_cube_mesh(self, size: float = 1.0, name: str = "cube") -> Mesh:
        """
        Create a simple cube mesh with colored faces

        Args:
            size: Size of the cube
            name: Name to store the mesh under

        Returns:
            The created mesh
        """
        half_size = size / 2.0

        # Define vertices for a cube
        vertices = np.array([
            # Front face
            [-half_size, -half_size,  half_size],
            [ half_size, -half_size,  half_size],
            [ half_size,  half_size,  half_size],
            [-half_size,  half_size,  half_size],
            # Back face
            [-half_size, -half_size, -half_size],
            [-half_size,  half_size, -half_size],
            [ half_size,  half_size, -half_size],
            [ half_size, -half_size, -half_size],
            # Top face
            [-half_size,  half_size, -half_size],
            [-half_size,  half_size,  half_size],
            [ half_size,  half_size,  half_size],
            [ half_size,  half_size, -half_size],
            # Bottom face
            [-half_size, -half_size, -half_size],
            [ half_size, -half_size, -half_size],
            [ half_size, -half_size,  half_size],
            [-half_size, -half_size,  half_size],
            # Right face
            [ half_size, -half_size, -half_size],
            [ half_size,  half_size, -half_size],
            [ half_size,  half_size,  half_size],
            [ half_size, -half_size,  half_size],
            # Left face
            [-half_size, -half_size, -half_size],
            [-half_size, -half_size,  half_size],
            [-half_size,  half_size,  half_size],
            [-half_size,  half_size, -half_size]
        ], dtype=np.float32)

        # Define texture coordinates (same for all faces)
        tex_coords = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Front
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Back
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Top
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Bottom
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Right
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]   # Left
        ], dtype=np.float32)

        # Define face colors (each face gets a different color)
        face_colors = [
            [1.0, 0.2, 0.2],  # Front - Red
            [0.2, 1.0, 0.2],  # Back - Green
            [0.2, 0.2, 1.0],  # Top - Blue
            [1.0, 1.0, 0.2],  # Bottom - Yellow
            [1.0, 0.2, 1.0],  # Right - Magenta
            [0.2, 1.0, 1.0]   # Left - Cyan
        ]

        colors = []
        for face_color in face_colors:
            colors.extend([face_color] * 4)  # 4 vertices per face
        colors = np.array(colors, dtype=np.float32)

        # Define indices for the cube faces
        indices = np.array([
            0,  1,  2,  0,  2,  3,    # Front
            4,  5,  6,  4,  6,  7,    # Back
            8,  9,  10, 8,  10, 11,   # Top
            12, 13, 14, 12, 14, 15,   # Bottom
            16, 17, 18, 16, 18, 19,   # Right
            20, 21, 22, 20, 22, 23    # Left
        ], dtype=np.uint32)

        mesh = Mesh(vertices, indices, tex_coords, colors)
        self.meshes[name] = mesh

        logger.info(f"Created cube mesh: {name}, size={size}")
        return mesh

    def create_colored_texture(self, width: int, height: int,
                              color: Tuple[int, int, int], name: str) -> Texture:
        """
        Create a simple colored texture

        Args:
            width: Texture width
            height: Texture height
            color: RGB color values (0-255)
            name: Name to store the texture under

        Returns:
            The created texture
        """
        data = np.full((height, width, 4), [*color, 255], dtype=np.uint8)
        texture = Texture(width, height, data)
        self.textures[name] = texture

        logger.info(f"Created colored texture: {name}, color={color}")
        return texture

    def render_mesh(self, mesh_name: str, texture_name: str = None):
        """
        Render a mesh with optional texture

        Args:
            mesh_name: Name of the mesh to render
            texture_name: Name of the texture to use (optional)
        """
        if mesh_name not in self.meshes:
            logger.warning(f"Mesh not found: {mesh_name}")
            return

        # Bind texture if provided
        if texture_name and texture_name in self.textures:
            self.textures[texture_name].bind(0)

        # Render the mesh
        self.meshes[mesh_name].draw()

    def create_chunk_mesh(self, mesh_data: MeshData, chunk_key: str) -> ChunkMesh:
        """
        Create and upload a chunk mesh to GPU

        Args:
            mesh_data: MeshData containing vertices and indices
            chunk_key: Unique identifier for the chunk (e.g., "chunk_0_0")

        Returns:
            Created ChunkMesh instance
        """
        # Clean up existing mesh for this chunk if it exists
        if chunk_key in self.chunk_meshes:
            self.remove_chunk_mesh(chunk_key)

        # Create new chunk mesh
        chunk_mesh = ChunkMesh(mesh_data)
        self.chunk_meshes[chunk_key] = chunk_mesh

        logger.debug(f"Created chunk mesh: {chunk_key}, "
                    f"{chunk_mesh.vertex_count} vertices, "
                    f"{chunk_mesh.face_count} faces, "
                    f"generation time: {chunk_mesh.generation_time*1000:.2f}ms")
        return chunk_mesh

    def update_chunk_mesh(self, mesh_data: MeshData, chunk_key: str) -> ChunkMesh:
        """
        Update an existing chunk mesh with new data

        Args:
            mesh_data: New mesh data
            chunk_key: Chunk identifier

        Returns:
            Updated ChunkMesh instance
        """
        # Remove old mesh
        self.remove_chunk_mesh(chunk_key)

        # Create new mesh
        return self.create_chunk_mesh(mesh_data, chunk_key)

    def remove_chunk_mesh(self, chunk_key: str):
        """
        Remove a chunk mesh and clean up its GPU resources

        Args:
            chunk_key: Chunk identifier to remove
        """
        if chunk_key in self.chunk_meshes:
            chunk_mesh = self.chunk_meshes[chunk_key]
            chunk_mesh.cleanup()
            del self.chunk_meshes[chunk_key]
            logger.debug(f"Removed chunk mesh: {chunk_key}")

    def render_chunk_mesh(self, chunk_key: str, texture_name: str = None):
        """
        Render a specific chunk mesh

        Args:
            chunk_key: Chunk identifier
            texture_name: Optional texture name to bind
        """
        if chunk_key not in self.chunk_meshes:
            logger.warning(f"Chunk mesh not found: {chunk_key}")
            return

        # Bind texture if provided
        if texture_name and texture_name in self.textures:
            self.textures[texture_name].bind(0)

        # Render the chunk mesh
        self.chunk_meshes[chunk_key].draw()

    def render_all_chunks(self, texture_name: str = None):
        """
        Render all loaded chunk meshes

        Args:
            texture_name: Optional texture name to bind
        """
        if not self.chunk_meshes:
            return

        # Bind texture if provided
        if texture_name and texture_name in self.textures:
            self.textures[texture_name].bind(0)

        # Render all chunk meshes
        for chunk_key, chunk_mesh in self.chunk_meshes.items():
            chunk_mesh.draw()

        logger.debug(f"Rendered {len(self.chunk_meshes)} chunk meshes")

    def create_texture_atlas(self, width: int, height: int, data: np.ndarray, name: str) -> Texture:
        """
        Create a texture atlas for block textures

        Args:
            width: Atlas width
            height: Atlas height
            data: RGBA pixel data
            name: Atlas name

        Returns:
            Created texture
        """
        texture = Texture(width, height, data)
        self.textures[name] = texture
        logger.info(f"Created texture atlas: {name}, size={width}x{height}")
        return texture

    def get_chunk_mesh_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded chunk meshes

        Returns:
            Dictionary with mesh statistics
        """
        total_vertices = sum(mesh.vertex_count for mesh in self.chunk_meshes.values())
        total_faces = sum(mesh.face_count for mesh in self.chunk_meshes.values())
        total_indices = sum(mesh.index_count for mesh in self.chunk_meshes.values())

        return {
            'chunk_count': len(self.chunk_meshes),
            'total_vertices': total_vertices,
            'total_faces': total_faces,
            'total_indices': total_indices,
            'avg_vertices_per_chunk': total_vertices // len(self.chunk_meshes) if self.chunk_meshes else 0,
            'avg_faces_per_chunk': total_faces // len(self.chunk_meshes) if self.chunk_meshes else 0
        }

    def begin_frame(self):
        """Called at the beginning of each frame"""
        pass

    def end_frame(self):
        """Called at the end of each frame"""
        pass

    def cleanup(self):
        """Clean up all GPU resources"""
        logger.info("Cleaning up renderer resources...")

        # Clean up regular meshes
        for mesh in self.meshes.values():
            mesh.cleanup()
        self.meshes.clear()

        # Clean up chunk meshes
        for chunk_mesh in self.chunk_meshes.values():
            chunk_mesh.cleanup()
        self.chunk_meshes.clear()

        # Clean up textures
        for texture in self.textures.values():
            texture.cleanup()
        self.textures.clear()

        # Clean up VBO pools
        if self._vbo_pool:
            from OpenGL.GL import glDeleteBuffers
            glDeleteBuffers(len(self._vbo_pool), self._vbo_pool)
            self._vbo_pool.clear()

        if self._vao_pool:
            from OpenGL.GL import glDeleteVertexArrays
            glDeleteVertexArrays(len(self._vao_pool), self._vao_pool)
            self._vao_pool.clear()

        if self._ebo_pool:
            from OpenGL.GL import glDeleteBuffers
            glDeleteBuffers(len(self._ebo_pool), self._ebo_pool)
            self._ebo_pool.clear()

        logger.info("Renderer cleanup complete")
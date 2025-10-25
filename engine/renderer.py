"""
Basic 3D rendering system for Pycraft
Handles VAO/VBO management, mesh creation, and rendering operations
"""

import logging
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
    Basic renderer for 3D meshes
    """

    def __init__(self):
        """Initialize renderer"""
        self.meshes: Dict[str, Mesh] = {}
        self.textures: Dict[str, Texture] = {}

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

    def begin_frame(self):
        """Called at the beginning of each frame"""
        pass

    def end_frame(self):
        """Called at the end of each frame"""
        pass

    def cleanup(self):
        """Clean up all GPU resources"""
        logger.info("Cleaning up renderer resources...")

        for mesh in self.meshes.values():
            mesh.cleanup()
        self.meshes.clear()

        for texture in self.textures.values():
            texture.cleanup()
        self.textures.clear()

        logger.info("Renderer cleanup complete")
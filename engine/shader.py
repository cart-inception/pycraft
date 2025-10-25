"""
Shader management system for modern OpenGL rendering
Handles compilation, linking, and uniform management
"""

import logging
from typing import Dict, Optional, Union
import os

from OpenGL.GL import (
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS,
    GL_LINK_STATUS, GL_TRUE, GL_FALSE, glCreateShader, glShaderSource,
    glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram,
    glGetProgramiv, glGetProgramInfoLog, glDeleteShader,
    glDeleteProgram, glUseProgram, glGetUniformLocation, glUniform1i,
    glUniform1f, glUniform3f, glUniformMatrix4fv
)
import numpy as np

from config import config

logger = logging.getLogger(__name__)


class ShaderError(Exception):
    """Exception raised when shader compilation/linking fails"""
    pass


class Shader:
    """
    Modern OpenGL shader program wrapper
    Handles vertex and fragment shaders with uniform management
    """

    def __init__(self, vertex_path: str, fragment_path: str):
        """
        Initialize shader program

        Args:
            vertex_path: Path to vertex shader file
            fragment_path: Path to fragment shader file
        """
        self.vertex_path = vertex_path
        self.fragment_path = fragment_path
        self.program_id: Optional[int] = None
        self.uniform_cache: Dict[str, int] = {}

        self._compile()

    def _compile(self):
        """Compile and link the shader program"""
        try:
            # Load shader sources
            vertex_source = self._load_shader_source(self.vertex_path)
            fragment_source = self._load_shader_source(self.fragment_path)

            # Compile shaders
            vertex_shader = self._compile_shader(vertex_source, GL_VERTEX_SHADER, self.vertex_path)
            fragment_shader = self._compile_shader(fragment_source, GL_FRAGMENT_SHADER, self.fragment_path)

            # Link program
            self.program_id = self._link_program(vertex_shader, fragment_shader)

            # Clean up individual shaders
            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)

            logger.info(f"Successfully compiled and linked shader program: {self.vertex_path} + {self.fragment_path}")

        except Exception as e:
            logger.error(f"Failed to compile shader program: {e}")
            raise ShaderError(f"Shader compilation failed: {e}")

    def _load_shader_source(self, path: str) -> str:
        """Load shader source code from file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shader file not found: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ShaderError(f"Failed to read shader file {path}: {e}")

    def _compile_shader(self, source: str, shader_type: int, path: str) -> int:
        """Compile individual shader"""
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)

        # Check compilation status
        success = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if success != GL_TRUE:
            info_log = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            shader_type_name = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
            raise ShaderError(f"{shader_type_name} shader compilation failed ({path}):\n{info_log}")

        return shader

    def _link_program(self, vertex_shader: int, fragment_shader: int) -> int:
        """Link shader program"""
        program = glCreateProgram()
        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)

        # Check linking status
        success = glGetProgramiv(program, GL_LINK_STATUS)
        if success != GL_TRUE:
            info_log = glGetProgramInfoLog(program)
            glDeleteProgram(program)
            raise ShaderError(f"Shader program linking failed:\n{info_log}")

        return program

    def use(self):
        """Use this shader program for rendering"""
        if self.program_id is None:
            raise ShaderError("Shader program not initialized")
        glUseProgram(self.program_id)

    def _get_uniform_location(self, name: str) -> int:
        """Get uniform location, with caching"""
        if name not in self.uniform_cache:
            if self.program_id is None:
                raise ShaderError("Shader program not initialized")
            location = glGetUniformLocation(self.program_id, name)
            self.uniform_cache[name] = location
        return self.uniform_cache[name]

    def set_int(self, name: str, value: int):
        """Set integer uniform"""
        location = self._get_uniform_location(name)
        if location >= 0:  # Only set if uniform exists
            glUniform1i(location, value)

    def set_float(self, name: str, value: float):
        """Set float uniform"""
        location = self._get_uniform_location(name)
        if location >= 0:
            glUniform1f(location, value)

    def set_vec3(self, name: str, x: float, y: float, z: float):
        """Set vec3 uniform"""
        location = self._get_uniform_location(name)
        if location >= 0:
            glUniform3f(location, x, y, z)

    def set_matrix4(self, name: str, matrix: np.ndarray):
        """
        Set 4x4 matrix uniform

        Args:
            name: Uniform name
            matrix: 4x4 numpy array
        """
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")

        location = self._get_uniform_location(name)
        if location >= 0:
            glUniformMatrix4fv(location, 1, GL_FALSE, matrix.astype(np.float32))

    def cleanup(self):
        """Clean up OpenGL resources"""
        if self.program_id is not None:
            glDeleteProgram(self.program_id)
            self.program_id = None
        self.uniform_cache.clear()


class ShaderManager:
    """
    Manager for shader programs with loading and caching
    """

    def __init__(self):
        self.shaders: Dict[str, Shader] = {}
        self.shaders_dir = config.get_shaders_dir()

    def load_shader(self, name: str, vertex_file: str = None, fragment_file: str = None) -> Shader:
        """
        Load or get cached shader

        Args:
            name: Shader name for caching
            vertex_file: Vertex shader filename (defaults to name + "_vertex.glsl")
            fragment_file: Fragment shader filename (defaults to name + "_fragment.glsl")
        """
        if name in self.shaders:
            return self.shaders[name]

        if vertex_file is None:
            vertex_file = f"{name}_vertex.glsl"
        if fragment_file is None:
            fragment_file = f"{name}_fragment.glsl"

        vertex_path = os.path.join(self.shaders_dir, vertex_file)
        fragment_path = os.path.join(self.shaders_dir, fragment_file)

        shader = Shader(vertex_path, fragment_path)
        self.shaders[name] = shader

        logger.info(f"Loaded and cached shader: {name}")
        return shader

    def get_shader(self, name: str) -> Optional[Shader]:
        """Get cached shader by name"""
        return self.shaders.get(name)

    def cleanup(self):
        """Clean up all shaders"""
        for shader in self.shaders.values():
            shader.cleanup()
        self.shaders.clear()
        logger.info("Cleaned up all shaders")

    def set_common_uniforms(self, shader: Shader, projection_matrix: np.ndarray,
                           view_matrix: np.ndarray, model_matrix: np.ndarray = None):
        """
        Set common uniforms used by most shaders

        Args:
            shader: Shader to set uniforms on
            projection_matrix: 4x4 projection matrix
            view_matrix: 4x4 view matrix
            model_matrix: 4x4 model matrix (identity if None)
        """
        shader.set_matrix4("uProjection", projection_matrix)
        shader.set_matrix4("uView", view_matrix)

        if model_matrix is not None:
            shader.set_matrix4("uModel", model_matrix)
        else:
            shader.set_matrix4("uModel", np.eye(4))

        # Set lighting uniforms
        shader.set_vec3("uAmbientColor", 0.4, 0.4, 0.5)  # Slightly blue ambient
        shader.set_vec3("uSunColor", 1.0, 0.95, 0.8)     # Warm sunlight
        shader.set_float("uTime", 0.0)  # Will be updated in game loop


# Global shader manager instance
shader_manager = ShaderManager()
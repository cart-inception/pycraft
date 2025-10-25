"""
FPS-style camera system for Pycraft
Handles camera movement, mouse look, and matrix generation
"""

import math
import logging
from typing import Tuple, Optional

import numpy as np
import pygame
from pygame.locals import *

from config import config
from engine.utils import create_perspective_matrix, create_look_at_matrix

logger = logging.getLogger(__name__)


class Camera:
    """
    First-person shooter style camera with mouse look and WASD movement
    """

    def __init__(self):
        """Initialize camera"""
        # Position and orientation
        self.position = np.array([0.0, 5.0, 10.0], dtype=np.float32)  # Start above ground
        self.yaw = -90.0  # Horizontal rotation (degrees)
        self.pitch = 0.0   # Vertical rotation (degrees)

        # Movement vectors
        self.forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Camera properties
        self.fov = config.FOV
        self.near_plane = config.NEAR_PLANE
        self.far_plane = config.FAR_PLANE

        # Movement state
        self.move_forward = False
        self.move_backward = False
        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False

        # Mouse state
        self.mouse_captured = False
        self.last_mouse_pos = None
        self.mouse_sensitivity = config.MOUSE_SENSITIVITY

        # Matrices (cached for performance)
        self.view_matrix = None
        self.projection_matrix = None
        self.view_dirty = True
        self.projection_dirty = True

        logger.info("Camera initialized")

    def set_position(self, x: float, y: float, z: float):
        """Set camera position"""
        self.position = np.array([x, y, z], dtype=np.float32)
        self.view_dirty = True

    def get_position(self) -> np.ndarray:
        """Get camera position"""
        return self.position.copy()

    def set_rotation(self, yaw: float, pitch: float):
        """
        Set camera rotation

        Args:
            yaw: Horizontal rotation in degrees
            pitch: Vertical rotation in degrees
        """
        self.yaw = yaw
        self.pitch = np.clip(pitch, -89.0, 89.0)  # Clamp to prevent gimbal lock
        self._update_vectors()
        self.view_dirty = True

    def get_rotation(self) -> Tuple[float, float]:
        """Get camera rotation as (yaw, pitch) in degrees"""
        return self.yaw, self.pitch

    def _update_vectors(self):
        """Update camera direction vectors based on rotation"""
        # Calculate new forward vector
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        self.forward[0] = math.cos(yaw_rad) * math.cos(pitch_rad)
        self.forward[1] = math.sin(pitch_rad)
        self.forward[2] = math.sin(yaw_rad) * math.cos(pitch_rad)
        self.forward = self.forward / np.linalg.norm(self.forward)

        # Calculate right and up vectors
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.cross(self.forward, world_up)
        self.right = self.right / np.linalg.norm(self.right)

        self.up = np.cross(self.right, self.forward)
        self.up = self.up / np.linalg.norm(self.up)

    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix"""
        if self.view_dirty or self.view_matrix is None:
            target = self.position + self.forward
            self.view_matrix = create_look_at_matrix(self.position, target, self.up)
            self.view_dirty = False
        return self.view_matrix

    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """Get the projection matrix"""
        if self.projection_dirty or self.projection_matrix is None:
            fov_rad = math.radians(self.fov)
            self.projection_matrix = create_perspective_matrix(
                fov_rad, aspect_ratio, self.near_plane, self.far_plane
            )
            self.projection_dirty = False
        return self.projection_matrix

    def handle_key_event(self, event: pygame.event.Event):
        """Handle keyboard events for movement"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                self.move_forward = True
            elif event.key == pygame.K_s:
                self.move_backward = True
            elif event.key == pygame.K_a:
                self.move_left = True
            elif event.key == pygame.K_d:
                self.move_right = True
            elif event.key == pygame.K_SPACE:
                self.move_up = True
            elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                self.move_down = True
            elif event.key == pygame.K_ESCAPE:
                self.toggle_mouse_capture()

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                self.move_forward = False
            elif event.key == pygame.K_s:
                self.move_backward = False
            elif event.key == pygame.K_a:
                self.move_left = False
            elif event.key == pygame.K_d:
                self.move_right = False
            elif event.key == pygame.K_SPACE:
                self.move_up = False
            elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                self.move_down = False

    def handle_mouse_event(self, event: pygame.event.Event):
        """Handle mouse events for camera rotation"""
        if not self.mouse_captured:
            return

        if event.type == pygame.MOUSEMOTION:
            if self.last_mouse_pos is not None:
                # Calculate mouse movement
                dx = event.pos[0] - self.last_mouse_pos[0]
                dy = event.pos[1] - self.last_mouse_pos[1]

                # Apply mouse sensitivity
                dx *= self.mouse_sensitivity
                dy *= self.mouse_sensitivity

                # Update rotation
                self.yaw += dx
                self.pitch -= dy  # Invert Y for natural feel

                # Clamp pitch to prevent gimbal lock
                self.pitch = np.clip(self.pitch, -89.0, 89.0)

                # Update vectors and mark view as dirty
                self._update_vectors()
                self.view_dirty = True

            self.last_mouse_pos = event.pos

            # Keep mouse centered
            center_x = config.WINDOW_WIDTH // 2
            center_y = config.WINDOW_HEIGHT // 2
            if abs(event.pos[0] - center_x) > 50 or abs(event.pos[1] - center_y) > 50:
                pygame.mouse.set_pos(center_x, center_y)
                self.last_mouse_pos = (center_x, center_y)

    def toggle_mouse_capture(self):
        """Toggle mouse capture mode"""
        self.mouse_captured = not self.mouse_captured

        if self.mouse_captured:
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)
            # Center mouse and set initial position
            center_x = config.WINDOW_WIDTH // 2
            center_y = config.WINDOW_HEIGHT // 2
            pygame.mouse.set_pos(center_x, center_y)
            self.last_mouse_pos = (center_x, center_y)
            logger.info("Mouse captured")
        else:
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
            self.last_mouse_pos = None
            logger.info("Mouse released")

    def update(self, delta_time: float):
        """
        Update camera position based on input

        Args:
            delta_time: Time since last frame in seconds
        """
        # Calculate movement speed (could be modified by sprinting, etc.)
        base_speed = config.PLAYER_SPEED

        # Frame-rate independent movement
        movement = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if self.move_forward:
            movement += self.forward * base_speed
        if self.move_backward:
            movement -= self.forward * base_speed
        if self.move_right:
            movement += self.right * base_speed
        if self.move_left:
            movement -= self.right * base_speed
        if self.move_up:
            movement += self.up * base_speed
        if self.move_down:
            movement -= self.up * base_speed

        # Normalize diagonal movement
        if np.linalg.norm(movement) > 0:
            movement = movement / np.linalg.norm(movement) * base_speed

        # Apply movement with delta time for frame-rate independence
        self.position += movement * delta_time
        self.view_dirty = True

    def get_frustum_corners(self, distance: float) -> np.ndarray:
        """
        Get the 8 corners of the camera frustum at a given distance
        Useful for culling and debugging

        Args:
            distance: Distance from camera to far plane

        Returns:
            8x3 array of frustum corner positions
        """
        aspect_ratio = config.WINDOW_WIDTH / config.WINDOW_HEIGHT
        fov_rad = math.radians(self.fov)

        # Calculate frustum dimensions at the given distance
        height = 2.0 * distance * math.tan(fov_rad / 2.0)
        width = height * aspect_ratio

        # Calculate frustum center
        center = self.position + self.forward * distance

        # Calculate right and up vectors scaled to frustum dimensions
        right_scaled = self.right * (width / 2.0)
        up_scaled = self.up * (height / 2.0)

        # Calculate the 8 corners
        corners = np.array([
            center + right_scaled + up_scaled,    # Top-right
            center - right_scaled + up_scaled,    # Top-left
            center - right_scaled - up_scaled,    # Bottom-left
            center + right_scaled - up_scaled,    # Bottom-right
            center + right_scaled + up_scaled,    # Near top-right (same as far for simplicity)
            center - right_scaled + up_scaled,    # Near top-left
            center - right_scaled - up_scaled,    # Near bottom-left
            center + right_scaled - up_scaled     # Near bottom-right
        ], dtype=np.float32)

        return corners

    def look_at(self, target: np.ndarray):
        """
        Make camera look at a specific target point

        Args:
            target: World position to look at
        """
        direction = target - self.position
        direction = direction / np.linalg.norm(direction)

        # Calculate pitch and yaw from direction
        self.pitch = math.degrees(math.asin(direction[1]))
        self.yaw = math.degrees(math.atan2(direction[2], direction[0]))

        self._update_vectors()
        self.view_dirty = True

    def set_fov(self, fov: float):
        """Set field of view in degrees"""
        self.fov = np.clip(fov, 1.0, 179.0)
        self.projection_dirty = True

    def get_fov(self) -> float:
        """Get field of view in degrees"""
        return self.fov

    def get_info(self) -> dict:
        """Get camera information for debugging"""
        return {
            'position': self.position.tolist(),
            'rotation': (self.yaw, self.pitch),
            'forward': self.forward.tolist(),
            'right': self.right.tolist(),
            'up': self.up.tolist(),
            'fov': self.fov,
            'mouse_captured': self.mouse_captured
        }
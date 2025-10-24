"""
Main entry point for Pycraft (Minecraft clone)
Initializes pygame, creates OpenGL context, and runs the main game loop
"""

import sys
import logging
import time
import math
from typing import Optional

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from config import config, logger
from engine.shader import shader_manager
from engine.utils import Timer, create_perspective_matrix, check_opengl_errors

class Game:
    """Main game class"""

    def __init__(self):
        """Initialize the game"""
        self.running: bool = False
        self.clock: Optional[pygame.time.Clock] = None
        self.window: Optional[pygame.Surface] = None
        self.delta_time: float = 0.0
        self.fps: float = 0.0

        # Test shader for basic rendering
        self.test_shader = None

        logger.info("Initializing Pycraft...")

    def initialize(self) -> bool:
        """
        Initialize pygame, OpenGL, and game systems

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize pygame
            self._init_pygame()

            # Initialize OpenGL
            self._init_opengl()

            # Initialize game systems
            self._init_systems()

            logger.info("Game initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize game: {e}")
            return False

    def _init_pygame(self):
        """Initialize pygame and create window"""
        logger.info("Initializing pygame...")

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Pycraft - Minecraft Clone")

        # Set display mode
        flags = pygame.DOUBLEBUF | pygame.OPENGL
        if config.FULLSCREEN:
            flags |= pygame.FULLSCREEN

        self.window = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
            flags
        )

        # Initialize clock
        self.clock = pygame.time.Clock()

        # Set mouse properties
        if config.MOUSE_LOCK:
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)

        logger.info(f"Pygame initialized: {config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")

    def _init_opengl(self):
        """Initialize OpenGL context"""
        logger.info("Initializing OpenGL...")

        # Set OpenGL attributes
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, config.OPENGL_VERSION_MAJOR)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, config.OPENGL_VERSION_MINOR)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

        # Enable multisample anti-aliasing
        if config.MSAA_SAMPLES > 0:
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
            pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, config.MSAA_SAMPLES)

        # Create OpenGL context
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)

        # Get OpenGL context info
        gl_version = glGetString(GL_VERSION).decode('utf-8')
        gl_renderer = glGetString(GL_RENDERER).decode('utf-8')
        gl_vendor = glGetString(GL_VENDOR).decode('utf-8')

        logger.info(f"OpenGL Version: {gl_version}")
        logger.info(f"OpenGL Renderer: {gl_renderer}")
        logger.info(f"OpenGL Vendor: {gl_vendor}")

        # Basic OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)

        # Set clear color (sky blue)
        glClearColor(0.5, 0.7, 1.0, 1.0)

        # Setup viewport
        glViewport(0, 0, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        # Check for OpenGL errors
        check_opengl_errors()

    def _init_systems(self):
        """Initialize game systems"""
        logger.info("Initializing game systems...")

        # Load test shader
        try:
            self.test_shader = shader_manager.load_shader("basic", "vertex.glsl", "fragment.glsl")
            logger.info("Test shader loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load test shader: {e}")
            raise

    def run(self):
        """Main game loop"""
        if not self.initialize():
            logger.error("Failed to initialize game, exiting...")
            return

        logger.info("Starting main game loop...")
        self.running = True

        try:
            while self.running:
                # Calculate delta time
                current_time = time.perf_counter()
                self.delta_time = current_time - last_time if 'last_time' in locals() else 0.016
                last_time = current_time

                # Handle events
                self._handle_events()

                # Update game logic
                self._update()

                # Render
                self._render()

                # Control frame rate and calculate FPS
                self.fps = self.clock.get_fps()
                self.clock.tick(config.FPS_TARGET)

        except KeyboardInterrupt:
            logger.info("Game interrupted by user")
        except Exception as e:
            logger.error(f"Game crashed: {e}")
            raise
        finally:
            self._cleanup()

    def _handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key == pygame.K_F3:
                    config.DEBUG_MODE = not config.DEBUG_MODE
                    logger.info(f"Debug mode: {'ON' if config.DEBUG_MODE else 'OFF'}")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Will be used for block breaking
                    pass
                elif event.button == 3:  # Right click
                    # Will be used for block placing
                    pass

            elif event.type == pygame.MOUSEMOTION:
                if config.MOUSE_LOCK:
                    # Will be used for camera rotation
                    pass

    def _update(self):
        """Update game logic"""
        # Placeholder for game logic updates
        # This will include player movement, physics, world updates, etc.
        pass

    def _render(self):
        """Render the game"""
        with Timer("Render Frame"):
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Render test shader (just clears screen for now)
            if self.test_shader:
                self.test_shader.use()
                # Set basic uniforms
                projection = create_perspective_matrix(
                    math.radians(config.FOV),
                    config.WINDOW_WIDTH / config.WINDOW_HEIGHT,
                    config.NEAR_PLANE,
                    config.FAR_PLANE
                )
                self.test_shader.set_matrix4("uProjection", projection)

            # Render debug info if enabled
            if config.DEBUG_MODE:
                self._render_debug_info()

            # Swap buffers
            pygame.display.flip()

            # Check for OpenGL errors
            check_opengl_errors()

    def _render_debug_info(self):
        """Render debug information overlay"""
        # This would render FPS, position, etc. using pygame drawing
        # For now, we'll just update the window title
        debug_info = f"Pycraft - FPS: {self.fps:.1f} | DEBUG MODE"
        pygame.display.set_caption(debug_info)

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        config.FULLSCREEN = not config.FULLSCREEN

        flags = pygame.DOUBLEBUF | pygame.OPENGL
        if config.FULLSCREEN:
            flags |= pygame.FULLSCREEN

        self.window = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT),
            flags
        )

        # Reset viewport
        glViewport(0, 0, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        logger.info(f"Fullscreen: {'ON' if config.FULLSCREEN else 'OFF'}")

    def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")

        # Cleanup shaders
        shader_manager.cleanup()

        # Cleanup pygame
        pygame.quit()

        logger.info("Cleanup complete")


def main():
    """Main entry point"""
    try:
        # Create and run game
        game = Game()
        game.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
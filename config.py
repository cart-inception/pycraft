"""
Central configuration file for Pycraft (Minecraft clone)
All game constants and settings should be defined here
"""

import os
from typing import Tuple


class Config:
    """Central configuration class for the game"""

    # ==========================================
    # DISPLAY & RENDERING SETTINGS
    # ==========================================
    WINDOW_WIDTH: int = 1280
    WINDOW_HEIGHT: int = 720
    FPS_TARGET: int = 60
    VSYNC: bool = True
    FULLSCREEN: bool = False
    MSAA_SAMPLES: int = 4  # Anti-aliasing samples

    # OpenGL settings
    OPENGL_VERSION_MAJOR: int = 3
    OPENGL_VERSION_MINOR: int = 3
    OPENGL_CORE_PROFILE: bool = True

    # Camera settings
    FOV: float = 70.0  # Field of view in degrees
    NEAR_PLANE: float = 0.1
    FAR_PLANE: float = 500.0
    MOUSE_SENSITIVITY: float = 0.2

    # ==========================================
    # WORLD & CHUNK SETTINGS
    # ==========================================
    # Chunk dimensions
    CHUNK_SIZE: int = 16  # X and Z dimensions
    CHUNK_HEIGHT: int = 256  # Y dimension (world height)

    # Render distance (in chunks)
    RENDER_DISTANCE: int = 8
    MAX_CHUNK_UPDATES_PER_FRAME: int = 4

    # World generation
    WORLD_SEED: int = None  # None = random seed
    TERRAIN_SCALE: float = 0.02
    TERRAIN_HEIGHT_SCALE: float = 32.0
    TERRAIN_OCTAVES: int = 4

    # ==========================================
    # PLAYER PHYSICS SETTINGS
    # ==========================================
    # Movement speeds (blocks per second)
    PLAYER_SPEED: float = 4.3
    PLAYER_SPRINT_SPEED: float = 5.6
    PLAYER_FLY_SPEED: float = 8.0
    JUMP_VELOCITY: float = 10.0
    GRAVITY: float = -32.0  # Blocks per second squared

    # Player dimensions
    PLAYER_HEIGHT: float = 1.8
    PLAYER_WIDTH: float = 0.6
    PLAYER_EYE_HEIGHT: float = 1.62  # Camera height from ground

    # ==========================================
    # GAMEPLAY SETTINGS
    # ==========================================
    REACH_DISTANCE: float = 5.0  # How far player can reach blocks
    BLOCK_BREAK_SPEED: float = 1.0  # Base block breaking speed
    CREATIVE_MODE: bool = False  # Enable creative mode features

    # Inventory settings
    MAX_INVENTORY_SIZE: int = 36
    HOTBAR_SIZE: int = 9
    MAX_STACK_SIZE: int = 64

    # ==========================================
    # PERFORMANCE SETTINGS
    # ==========================================
    # Threading
    MESH_GENERATION_THREADS: int = 4
    ASSET_LOADING_THREADS: int = 2

    # Memory management
    MAX_LOADED_CHUNKS: int = 1000
    UNLOAD_DISTANCE: int = RENDER_DISTANCE + 4
    CHUNK_CACHE_SIZE: int = 500

    # ==========================================
    # DEBUG & DEVELOPMENT SETTINGS
    # ==========================================
    DEBUG_MODE: bool = True  # Enable debug overlay and features
    SHOW_CHUNK_BORDERS: bool = False
    SHOW_COLLISION_BOXES: bool = False
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR

    # Profiling
    ENABLE_PROFILING: bool = False
    PROFILING_SAMPLE_RATE: float = 1.0  # Sample every N seconds

    # ==========================================
    # FILE PATHS
    # ==========================================
    @staticmethod
    def get_project_root() -> str:
        """Get the root directory of the project"""
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def get_saves_dir() -> str:
        """Get the saves directory"""
        saves_dir = os.path.join(Config.get_project_root(), "saves")
        os.makedirs(saves_dir, exist_ok=True)
        return saves_dir

    @staticmethod
    def get_resources_dir() -> str:
        """Get the resources directory"""
        resources_dir = os.path.join(Config.get_project_root(), "resources")
        os.makedirs(resources_dir, exist_ok=True)
        return resources_dir

    @staticmethod
    def get_textures_dir() -> str:
        """Get the textures directory"""
        textures_dir = os.path.join(Config.get_resources_dir(), "textures")
        os.makedirs(textures_dir, exist_ok=True)
        return textures_dir

    @staticmethod
    def get_shaders_dir() -> str:
        """Get the shaders directory"""
        shaders_dir = os.path.join(Config.get_project_root(), "shaders")
        os.makedirs(shaders_dir, exist_ok=True)
        return shaders_dir

    @staticmethod
    def get_audio_dir() -> str:
        """Get the audio directory"""
        audio_dir = os.path.join(Config.get_project_root(), "audio")
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir

    # ==========================================
    # BLOCK REGISTRY (Basic blocks for now)
    # ==========================================
    # Block IDs (0 is always air)
    BLOCK_AIR: int = 0
    BLOCK_STONE: int = 1
    BLOCK_GRASS: int = 2
    BLOCK_DIRT: int = 3
    BLOCK_BEDROCK: int = 4
    BLOCK_WOOD: int = 5
    BLOCK_PLANKS: int = 6
    BLOCK_LEAVES: int = 7
    BLOCK_GLASS: int = 8
    BLOCK_COBBLESTONE: int = 9
    BLOCK_SAND: int = 10
    BLOCK_WATER: int = 11
    BLOCK_COAL_ORE: int = 12
    BLOCK_IRON_ORE: int = 13
    BLOCK_GOLD_ORE: int = 14
    BLOCK_DIAMOND_ORE: int = 15

    # ==========================================
    # TEXTURE ATLAS SETTINGS
    # ==========================================
    TEXTURE_SIZE: int = 16  # Individual texture size
    ATLAS_SIZE: int = 8  # Atlas grid size (8x8 = 64 textures)
    ATLAS_TOTAL_SIZE: int = TEXTURE_SIZE * ATLAS_SIZE  # 128x128 pixels

    # ==========================================
    # INPUT SETTINGS
    # ==========================================
    # Key bindings
    KEY_FORWARD: str = "w"
    KEY_BACKWARD: str = "s"
    KEY_LEFT: str = "a"
    KEY_RIGHT: str = "d"
    KEY_JUMP: str = "space"
    KEY_SNEAK: str = "shift"
    KEY_SPRINT: str = "ctrl"
    KEY_FLY: str = "f"
    KEY_INVENTORY: str = "e"
    KEY_DEBUG: str = "f3"
    KEY_ESCAPE: str = "escape"

    # Mouse settings
    MOUSE_LOCK: bool = True  # Lock mouse to window
    MOUSE_INVERT: bool = False  # Invert Y axis

    # ==========================================
    # VALIDATION HELPERS
    # ==========================================
    @staticmethod
    def validate_block_id(block_id: int) -> bool:
        """Check if a block ID is valid"""
        return 0 <= block_id <= Config.BLOCK_DIAMOND_ORE

    @staticmethod
    def get_window_size() -> Tuple[int, int]:
        """Get the window size as a tuple"""
        return (Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)

    @staticmethod
    def get_chunk_dimensions() -> Tuple[int, int, int]:
        """Get chunk dimensions as a tuple"""
        return (Config.CHUNK_SIZE, Config.CHUNK_HEIGHT, Config.CHUNK_SIZE)

    @staticmethod
    def get_player_dimensions() -> Tuple[float, float, float]:
        """Get player dimensions as a tuple"""
        return (Config.PLAYER_WIDTH, Config.PLAYER_HEIGHT, Config.PLAYER_EYE_HEIGHT)


# Create a global config instance for easy access
config = Config()


# ==========================================
# LOGGING CONFIGURATION
# ==========================================
def setup_logging():
    """Setup logging based on config settings"""
    import logging

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }

    log_level = level_map.get(Config.LOG_LEVEL.upper(), logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.get_project_root(), 'game.log')),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


# Initialize logger when config is imported
logger = setup_logging()
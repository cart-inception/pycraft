"""
Tests for configuration system
"""

import pytest
import os
import tempfile

from config import config, Config


class TestConfig:
    """Test configuration class and values"""

    def test_config_constants(self):
        """Test that required config constants exist and have valid values"""
        # Display settings
        assert hasattr(config, 'WINDOW_WIDTH')
        assert config.WINDOW_WIDTH > 0
        assert hasattr(config, 'WINDOW_HEIGHT')
        assert config.WINDOW_HEIGHT > 0
        assert hasattr(config, 'FPS_TARGET')
        assert config.FPS_TARGET > 0

        # World settings
        assert hasattr(config, 'CHUNK_SIZE')
        assert config.CHUNK_SIZE > 0
        assert hasattr(config, 'CHUNK_HEIGHT')
        assert config.CHUNK_HEIGHT > 0
        assert hasattr(config, 'RENDER_DISTANCE')
        assert config.RENDER_DISTANCE > 0

        # Player settings
        assert hasattr(config, 'PLAYER_SPEED')
        assert config.PLAYER_SPEED > 0
        assert hasattr(config, 'GRAVITY')
        assert config.GRAVITY < 0  # Should be negative

        # Block IDs
        assert hasattr(config, 'BLOCK_AIR')
        assert config.BLOCK_AIR == 0
        assert hasattr(config, 'BLOCK_STONE')
        assert config.BLOCK_STONE > 0

    def test_path_methods(self):
        """Test path helper methods"""
        project_root = Config.get_project_root()
        assert os.path.exists(project_root)
        assert os.path.isdir(project_root)

        saves_dir = Config.get_saves_dir()
        assert os.path.exists(saves_dir)
        assert os.path.isdir(saves_dir)

        resources_dir = Config.get_resources_dir()
        assert os.path.exists(resources_dir)
        assert os.path.isdir(resources_dir)

        textures_dir = Config.get_textures_dir()
        assert os.path.exists(textures_dir)
        assert os.path.isdir(textures_dir)

        shaders_dir = Config.get_shaders_dir()
        assert os.path.exists(shaders_dir)
        assert os.path.isdir(shaders_dir)

    def test_validation_helpers(self):
        """Test validation helper methods"""
        # Test block ID validation
        assert Config.validate_block_id(config.BLOCK_AIR)
        assert Config.validate_block_id(config.BLOCK_STONE)
        assert Config.validate_block_id(config.BLOCK_DIAMOND_ORE)
        assert not Config.validate_block_id(-1)
        assert not Config.validate_block_id(config.BLOCK_DIAMOND_ORE + 1)

        # Test tuple getters
        window_size = config.get_window_size()
        assert isinstance(window_size, tuple)
        assert len(window_size) == 2
        assert window_size == (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        chunk_dims = config.get_chunk_dimensions()
        assert isinstance(chunk_dims, tuple)
        assert len(chunk_dims) == 3
        assert chunk_dims == (config.CHUNK_SIZE, config.CHUNK_HEIGHT, config.CHUNK_SIZE)

        player_dims = config.get_player_dimensions()
        assert isinstance(player_dims, tuple)
        assert len(player_dims) == 3

    def test_config_values_type(self):
        """Test that config values have correct types"""
        # Numeric values
        assert isinstance(config.WINDOW_WIDTH, int)
        assert isinstance(config.WINDOW_HEIGHT, int)
        assert isinstance(config.FPS_TARGET, int)
        assert isinstance(config.CHUNK_SIZE, int)
        assert isinstance(config.CHUNK_HEIGHT, int)
        assert isinstance(config.RENDER_DISTANCE, int)

        # Float values
        assert isinstance(config.FOV, float)
        assert isinstance(config.PLAYER_SPEED, float)
        assert isinstance(config.GRAVITY, float)
        assert isinstance(config.REACH_DISTANCE, float)

        # Boolean values
        assert isinstance(config.FULLSCREEN, bool)
        assert isinstance(config.DEBUG_MODE, bool)
        assert isinstance(config.MOUSE_LOCK, bool)

    def test_block_id_constants(self):
        """Test block ID constants are sequential and valid"""
        block_ids = [
            config.BLOCK_AIR,
            config.BLOCK_STONE,
            config.BLOCK_GRASS,
            config.BLOCK_DIRT,
            config.BLOCK_BEDROCK,
            config.BLOCK_WOOD,
            config.BLOCK_PLANKS,
            config.BLOCK_LEAVES,
            config.BLOCK_GLASS,
            config.BLOCK_COBBLESTONE,
            config.BLOCK_SAND,
            config.BLOCK_WATER,
            config.BLOCK_COAL_ORE,
            config.BLOCK_IRON_ORE,
            config.BLOCK_GOLD_ORE,
            config.BLOCK_DIAMOND_ORE
        ]

        # Should be 0-15 for 16 block types
        for i, block_id in enumerate(block_ids):
            assert block_id == i, f"Block ID {block_id} should be {i}"

        # Should be unique
        assert len(block_ids) == len(set(block_ids))
"""
Pytest configuration and fixtures for Pycraft testing
"""

import pytest
import numpy as np
import tempfile
import os
import shutil

from config import config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    # Override some config values for testing
    original_values = {
        'WINDOW_WIDTH': config.WINDOW_WIDTH,
        'WINDOW_HEIGHT': config.WINDOW_HEIGHT,
        'DEBUG_MODE': config.DEBUG_MODE,
        'CHUNK_SIZE': config.CHUNK_SIZE,
        'CHUNK_HEIGHT': config.CHUNK_HEIGHT
    }

    # Set test values
    config.WINDOW_WIDTH = 800
    config.WINDOW_HEIGHT = 600
    config.DEBUG_MODE = True
    config.CHUNK_SIZE = 4  # Smaller for faster tests
    config.CHUNK_HEIGHT = 64

    yield config

    # Restore original values
    for key, value in original_values.items():
        setattr(config, key, value)


@pytest.fixture
def sample_matrix():
    """Create a sample 4x4 matrix for testing"""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)


@pytest.fixture
def sample_vector():
    """Create a sample 3D vector for testing"""
    return np.array([1.0, 2.0, 3.0], dtype=np.float32)
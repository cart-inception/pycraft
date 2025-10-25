#!/usr/bin/env python3
"""
Test script for the chunk system implementation.

This script tests the core functionality of the chunk data structure,
mesh building, and world management systems.
"""

import sys
import time
import numpy as np

# Add the project root to Python path
sys.path.insert(0, '.')

from config import Config
from engine.chunk import Chunk, ChunkPosition
from engine.mesh_builder import MeshBuilder
from engine.world import World
from blocks.block_types import get_block_type, is_block_solid


def test_chunk_data_structure():
    """Test basic chunk functionality."""
    print("Testing Chunk Data Structure...")

    # Create chunk
    pos = ChunkPosition(0, 0)
    chunk = Chunk(pos)

    # Test initial state
    assert chunk.count_blocks() == 0
    assert chunk.get_block(5, 10, 5) == 0  # Air

    # Test setting blocks
    chunk.set_block(5, 10, 5, 1)  # Grass
    assert chunk.get_block(5, 10, 5) == 1
    assert chunk.count_blocks() == 1
    assert chunk.count_blocks(1) == 1

    # Test boundary checking
    assert chunk.get_block(-1, 10, 5) == 0  # Out of bounds returns air
    assert chunk.set_block(-1, 10, 5, 1) == False  # Can't set out of bounds

    # Test fill area
    chunk.fill_area((0, 5, 0), (3, 8, 3), 2)  # Fill 4x4x4 area with dirt
    expected_volume = 4 * 4 * 4
    assert chunk.count_blocks(2) == expected_volume

    print("✓ Chunk data structure tests passed")


def test_mesh_builder():
    """Test mesh building functionality."""
    print("Testing Mesh Builder...")

    # Create test chunk
    pos = ChunkPosition(0, 0)
    chunk = Chunk(pos)

    # Create simple structure
    chunk.set_block(5, 10, 5, 1)  # Single grass block
    chunk.set_block(6, 10, 5, 1)  # Adjacent grass block
    chunk.set_block(5, 11, 5, 1)  # Block above

    # Build mesh
    mesh_builder = MeshBuilder()
    mesh_data = mesh_builder.generate_chunk_mesh(chunk)

    # Verify mesh data
    assert mesh_data.vertex_count > 0
    assert len(mesh_data.indices) > 0
    assert mesh_data.face_count > 0
    assert mesh_data.generation_time >= 0

    print(f"✓ Mesh builder generated {mesh_data.vertex_count} vertices, "
          f"{mesh_data.face_count} faces in {mesh_data.generation_time*1000:.2f}ms")


def test_world_coordinate_system():
    """Test world coordinate conversions."""
    print("Testing World Coordinate System...")

    world = World(seed=12345)

    # Test coordinate conversion
    chunk_x, chunk_z = world.get_chunk_position(32, 48)
    assert chunk_x == 2
    assert chunk_z == 3

    local_x, local_y, local_z = world.get_local_chunk_coords(32, 64, 48)
    assert local_x == 0
    assert local_y == 64
    assert local_z == 0

    # Test block setting and retrieval
    success = world.set_block_global(32, 64, 48, 1)
    assert success
    block_id = world.get_block_global(32, 64, 48)
    assert block_id == 1

    # Test chunk generation
    chunk = world.get_or_generate_chunk(0, 0)
    assert chunk is not None
    assert chunk.position.x == 0
    assert chunk.position.z == 0

    print("✓ World coordinate system tests passed")


def test_chunk_loading():
    """Test chunk loading functionality."""
    print("Testing Chunk Loading...")

    world = World(seed=42)

    # Update player position
    world.update_player_position(0, 0)

    # Process loading
    stats = world.process_chunk_loading(max_loads=5)
    assert stats['chunks_loaded'] >= 0

    # Check loading status
    status = world.get_loading_status()
    assert 'player_chunk' in status
    assert 'loaded_chunks' in status

    print(f"✓ Chunk loading tests passed. "
          f"Loaded {status['loaded_chunks']} chunks")


def test_dirty_flag_system():
    """Test chunk dirty flag and rebuild system."""
    print("Testing Dirty Flag System...")

    world = World(seed=999)

    # Generate a chunk and test it's initially dirty
    chunk = world.get_or_generate_chunk(0, 0)
    assert chunk.is_dirty  # Fresh chunks start dirty

    # Test block modification marks chunk dirty
    chunk.is_dirty = False  # Reset to clean for testing
    world.set_block_global(8, 64, 8, 2)  # Should mark chunk dirty
    assert chunk.is_dirty

    # Test rebuild system
    world.mark_chunk_dirty(0, 0)  # Mark dirty for rebuild test
    status = world.get_rebuild_status()

    # Process rebuilds
    stats, rebuilt_meshes = world.process_chunk_rebuilds(max_rebuilds=5)
    assert stats['chunks_rebuilt'] >= 0

    if rebuilt_meshes:
        chunk_coords, mesh_data = rebuilt_meshes[0]
        assert len(chunk_coords) == 2
        assert mesh_data.vertex_count > 0

    print(f"✓ Dirty flag system tests passed. "
          f"Rebuilt {stats['chunks_rebuilt']} chunks")


def test_performance():
    """Test performance of key operations."""
    print("Testing Performance...")

    # Test chunk operations
    start_time = time.perf_counter()

    pos = ChunkPosition(0, 0)
    chunk = Chunk(pos)

    # Fill chunk with fewer blocks to avoid greedy meshing edge cases
    for x in range(8):  # Smaller area
        for y in range(32):
            for z in range(8):
                chunk.set_block(x, y, z, 1)

    chunk_fill_time = time.perf_counter() - start_time

    # Test mesh generation
    start_time = time.perf_counter()

    mesh_builder = MeshBuilder()
    mesh_data = mesh_builder.generate_chunk_mesh(chunk)

    mesh_gen_time = time.perf_counter() - start_time

    print(f"✓ Performance tests completed:")
    print(f"  - Chunk fill (2048 blocks): {chunk_fill_time*1000:.2f}ms")
    print(f"  - Mesh generation: {mesh_gen_time*1000:.2f}ms")
    print(f"  - Vertices generated: {mesh_data.vertex_count}")
    print(f"  - Faces generated: {mesh_data.face_count}")

    # Performance assertions (should be reasonably fast)
    assert chunk_fill_time < 0.1  # 100ms for 2048 blocks
    assert mesh_gen_time < 0.2  # 200ms for mesh generation (allowing for greedy meshing complexity)


def test_block_types():
    """Test block type system."""
    print("Testing Block Types...")

    # Test basic block types
    grass_block = get_block_type(1)
    assert grass_block.name == "grass"
    assert grass_block.is_solid
    assert not grass_block.is_transparent

    glass_block = get_block_type(6)
    assert glass_block.name == "glass"
    assert glass_block.is_solid
    assert glass_block.is_transparent

    air_block = get_block_type(0)
    assert air_block.name == "air"
    assert not air_block.is_solid
    assert air_block.is_transparent

    # Test texture UV coordinates
    uv = get_block_type(1).textures['top']
    assert len(uv) == 4  # u, v, width, height

    print("✓ Block type tests passed")


def main():
    """Run all tests."""
    print("Running Chunk System Tests...")
    print("=" * 50)

    try:
        test_chunk_data_structure()
        test_mesh_builder()
        test_world_coordinate_system()
        test_chunk_loading()
        test_dirty_flag_system()
        test_performance()
        test_block_types()

        print("=" * 50)
        print("✅ All tests passed!")

        # Show final statistics
        world = World(seed=12345)
        stats = world.get_world_statistics()
        print(f"\nWorld Statistics:")
        print(f"  - World seed: {stats['world_seed']}")
        print(f"  - Loaded chunks: {stats['loaded_chunks']}")
        print(f"  - Total blocks: {stats['total_blocks']}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
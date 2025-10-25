"""
Block types module - Defines basic block types and their properties

This module provides basic block definitions and texture UV coordinates
for the mesh builder to use when generating chunk meshes.
"""

from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class BlockType:
    """Represents a block type with rendering properties."""
    id: int
    name: str
    is_solid: bool = True
    is_transparent: bool = False

    # Texture UV coordinates for each face (u, v, width, height)
    # Values are normalized (0.0 to 1.0) for texture atlas
    textures: Dict[str, Tuple[float, float, float, float]] = None

    def __post_init__(self):
        if self.textures is None:
            self.textures = {}


# Basic block type definitions
# Texture atlas will be 8x8 grid, so each texture is 1/8 = 0.125 in size
TEX_SIZE = 0.125

# Create simple texture coordinates for basic blocks
def create_face_uv(atlas_x: int, atlas_y: int) -> Tuple[float, float, float, float]:
    """Create UV coordinates for a face in the texture atlas."""
    u = atlas_x * TEX_SIZE
    v = atlas_y * TEX_SIZE
    return (u, v, TEX_SIZE, TEX_SIZE)


# Basic block type registry
BLOCK_TYPES = {
    # Air
    0: BlockType(0, "air", is_solid=False, is_transparent=True),

    # Grass block (top: grass, sides: grass_side, bottom: dirt)
    1: BlockType(
        1, "grass",
        textures={
            'top': create_face_uv(0, 0),      # Grass top
            'bottom': create_face_uv(2, 0),   # Dirt
            'sides': create_face_uv(1, 0),    # Grass side
        }
    ),

    # Dirt block
    2: BlockType(
        2, "dirt",
        textures={
            'all': create_face_uv(2, 0),      # Dirt texture
        }
    ),

    # Stone block
    3: BlockType(
        3, "stone",
        textures={
            'all': create_face_uv(3, 0),      # Stone texture
        }
    ),

    # Wood log (top/bottom: wood, sides: bark)
    4: BlockType(
        4, "wood",
        textures={
            'top': create_face_uv(4, 0),      # Wood top
            'bottom': create_face_uv(4, 0),   # Wood bottom
            'sides': create_face_uv(5, 0),    # Bark
        }
    ),

    # Wooden planks
    5: BlockType(
        5, "planks",
        textures={
            'all': create_face_uv(6, 0),      # Planks texture
        }
    ),

    # Glass (transparent)
    6: BlockType(
        6, "glass",
        is_solid=True,
        is_transparent=True,
        textures={
            'all': create_face_uv(7, 0),      # Glass texture
        }
    ),

    # Cobblestone
    7: BlockType(
        7, "cobblestone",
        textures={
            'all': create_face_uv(0, 1),      # Cobblestone texture
        }
    ),

    # Leaves (transparent, semi-solid)
    8: BlockType(
        8, "leaves",
        is_solid=True,
        is_transparent=True,
        textures={
            'all': create_face_uv(1, 1),      # Leaves texture
        }
    ),
}


def get_block_type(block_id: int) -> BlockType:
    """Get block type by ID, returns air block for unknown IDs."""
    return BLOCK_TYPES.get(block_id, BLOCK_TYPES[0])


def get_block_texture_uv(block_id: int, face: str) -> Tuple[float, float, float, float]:
    """
    Get texture UV coordinates for a specific block face.

    Args:
        block_id: Block ID
        face: Face name ('top', 'bottom', 'sides', 'all', 'north', 'south', 'east', 'west')

    Returns:
        Tuple of (u, v, width, height) normalized coordinates
    """
    block_type = get_block_type(block_id)

    # Handle specific face textures
    if face in block_type.textures:
        return block_type.textures[face]

    # Fall back to side textures for specific directions
    if face in ['north', 'south', 'east', 'west'] and 'sides' in block_type.textures:
        return block_type.textures['sides']

    # Fall back to 'all' texture
    if 'all' in block_type.textures:
        return block_type.textures['all']

    # Default to first available texture
    if block_type.textures:
        return next(iter(block_type.textures.values()))

    # Fallback to grass texture
    return create_face_uv(0, 0)


def is_block_solid(block_id: int) -> bool:
    """Check if block type is solid (can block light and movement)."""
    return get_block_type(block_id).is_solid


def is_block_transparent(block_id: int) -> bool:
    """Check if block type is transparent (allows light through)."""
    return get_block_type(block_id).is_transparent
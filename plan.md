

# Minecraft Clone in Python: Development Plan

## Overview
This plan outlines how to create a modular, 3D Minecraft clone in Python with mining, crafting, and friendly mobs, using procedurally generated graphics.

## Technology Stack

### Core Libraries
- **Pygame**: Main game engine and window management
- **PyOpenGL**: 3D rendering with modern OpenGL (3.3+ core profile recommended)
- **NumPy**: Mathematical operations for 3D transformations and chunk data storage
- **Noise**: Perlin/Simplex noise for terrain generation
- **Pillow (PIL)**: Texture generation and manipulation

### Additional Dependencies
- **PyGLM** or **glm**: Better matrix/vector operations than raw NumPy (OpenGL-style math)
- **moderngl** (Optional): Consider as alternative to PyOpenGL for better performance and pythonic API
- **pickle** or **json**: For save/load system (standard library)
- **dataclasses**: For structured data (standard library, Python 3.7+)

### Performance Considerations
⚠️ **CRITICAL**: Python + OpenGL for voxel games has significant performance limitations:
- Consider using **Cython** or **Numba** for hot paths (mesh generation, collision detection)
- May need to implement mesh batching/instancing early, not in optimization phase
- Alternative: Consider **Pyglet** or **ModernGL** instead of Pygame+PyOpenGL for better OpenGL integration

### Project Structure
```
minecraft_clone/
├── main.py                 # Entry point
├── config.py               # Game settings and constants
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_world.py
│   ├── test_blocks.py
│   ├── test_crafting.py
│   └── test_entities.py
├── engine/                 # Core game engine
│   ├── __init__.py
│   ├── renderer.py         # OpenGL rendering and shader management
│   ├── camera.py           # Player camera with frustum
│   ├── world.py            # World management and chunk system
│   ├── chunk.py            # Individual chunk class ⚠️ MISSING
│   ├── mesh_builder.py     # Mesh generation for chunks ⚠️ MISSING
│   ├── input_handler.py    # Keyboard/mouse input ⚠️ MISSING
│   └── utils.py            # Utility functions
├── blocks/                 # Block system
│   ├── __init__.py
│   ├── block.py            # Base block class and block registry
│   ├── block_types.py      # Specific block implementations
│   └── block_textures.py   # Procedural texture generation
├── entities/               # Mobs and entities
│   ├── __init__.py
│   ├── entity.py           # Base entity class
│   ├── player.py           # Player entity (separate from camera) ⚠️ MISSING
│   ├── mobs.py             # Cow, chicken implementations
│   └── ai.py               # Basic AI behaviors
├── ui/                     # User interface
│   ├── __init__.py
│   ├── hud.py              # Heads-up display
│   ├── inventory.py        # Inventory system
│   ├── hotbar.py           # Quick access bar ⚠️ MISSING
│   └── crafting.py         # Crafting interface
├── mechanics/              # Game mechanics
│   ├── __init__.py
│   ├── mining.py           # Mining mechanics
│   ├── crafting.py         # Crafting recipes and logic
│   ├── physics.py          # Basic physics (gravity, collisions)
│   └── raycast.py          # Block selection raycasting ⚠️ MISSING
├── world/                  # World generation (NEW FOLDER)
│   ├── __init__.py
│   ├── terrain_generator.py # Terrain generation logic
│   ├── biomes.py           # Biome definitions
│   └── structures.py       # Trees, caves, etc.
├── shaders/                # GLSL shader files ⚠️ MISSING
│   ├── vertex.glsl         # Vertex shader
│   ├── fragment.glsl       # Fragment shader
│   └── skybox.glsl         # Skybox shaders (optional)
├── audio/                  # Audio system ⚠️ MISSING
│   ├── __init__.py
│   └── sound_generator.py  # Procedural sound effects
├── saves/                  # Save game directory (created at runtime)
└── resources/              # Generated/cached resources
    ├── textures/           # Procedurally generated textures (cached)
    └── models/             # Simple 3D models (optional)
```

**Key Structural Changes:**
- Added `world/` folder for terrain generation logic (better separation of concerns)
- Added `shaders/` folder - essential for modern OpenGL
- Added `tests/` folder - critical for maintainability
- Added missing modules: `chunk.py`, `mesh_builder.py`, `input_handler.py`, `player.py`, `hotbar.py`, `raycast.py`
- Added `audio/` folder for sound generation

## Development Phases

### Phase 0: Project Setup & Foundation (Week 1)
⚠️ **NEW PHASE** - Critical foundation work

#### 0.1 Environment Setup
- Set up virtual environment
- Install and test all dependencies
- Create requirements.txt
- Set up version control (git)
- Create .gitignore (exclude saves/, __pycache__, *.pyc, resources/textures/)

#### 0.2 Core Architecture
- Design and implement configuration system (config.py)
- Create shader loading system
- Write basic vertex and fragment shaders
- Set up logging system for debugging
- Implement basic error handling framework

#### 0.3 Testing Framework
- Set up pytest or unittest
- Create test structure
- Write example tests for utility functions

### Phase 1: Core Engine (Weeks 2-3)

#### 1.1 Basic 3D Rendering
- Set up Pygame window with OpenGL context (use OpenGL 3.3+ core profile)
- Implement shader program loading and compilation
- Create Vertex Array Objects (VAO) and Vertex Buffer Objects (VBO) system
- Implement basic camera with movement (WASD + mouse look)
- Add mouse capture/release (ESC key)
- Create simple cube rendering as proof of concept
- Implement basic lighting (directional light)

**⚠️ CRITICAL:** Use modern OpenGL (VBOs/VAOs), not immediate mode (glBegin/glEnd)

#### 1.2 Chunk System
- Design chunk data structure (16x16x16 or 16x256x16)
- Implement chunk class with block storage (use NumPy 3D arrays)
- Create chunk mesh generation (greedy meshing algorithm for performance)
- Implement face culling (don't render faces between solid blocks)
- Add chunk loading/unloading based on player position
- Implement chunk manager for neighboring chunk access

**⚠️ PERFORMANCE:** Mesh generation is the bottleneck - consider Numba/Cython here

#### 1.3 World Generation
- Implement 2D Perlin/Simplex noise for height maps
- Create basic terrain generator (hills, valleys)
- Add multiple noise octaves for detail
- Generate chunks on-demand as player moves
- Implement basic block types (grass, dirt, stone, bedrock)
- Add simple biome system (plains, forest)

#### 1.4 Physics & Collision
- Implement AABB (Axis-Aligned Bounding Box) collision detection
- Add gravity and jumping
- Implement player physics (walking, jumping, flying mode for debug)
- Create collision response system
- Add ground detection and slope handling

**⚠️ PITFALL:** Collision detection can be tricky at chunk boundaries

### Phase 2: Block Interaction (Weeks 4-5)

#### 2.1 Block Selection
- Implement raycasting for block selection
- Add block highlighting (wireframe or transparent overlay)
- Show selected block face
- Implement maximum reach distance

**⚠️ CRITICAL:** Raycasting algorithm needs to be robust (DDA or step-based)

#### 2.2 Block Types & Properties
- Implement base block class with properties:
  - Solid/transparent/liquid
  - Hardness (mining time)
  - Tool requirement
  - Drop items
  - Light emission (for later)
  - Custom rendering (for later - plants, water)
- Create block registry system
- Add different block types:
  - Solid: stone, dirt, grass, wood, planks, cobblestone
  - Transparent: glass, leaves
  - Plants: flowers, grass, saplings (need custom rendering)
  - Special: crafting table, furnace (interactive blocks)

#### 2.3 Mining Mechanics
- Implement block breaking with progress indicator
- Add tool system (hand, pickaxe, shovel, axe)
- Create tool effectiveness system (wood < stone < iron < diamond)
- Implement block dropping and item entities
- Add item collection (magnetic pull toward player)
- Create basic inventory system (list-based storage)

#### 2.4 Block Placement
- Implement block placement from inventory
- Add placement restrictions (can't place inside player)
- Handle placement on different faces
- Add block placement sound/visual feedback

### Phase 3: UI & Inventory (Week 6)

#### 3.1 HUD System
- Create crosshair
- Add hotbar (9 slots at bottom)
- Show selected hotbar slot
- Display item counts
- Add health and hunger bars (even if not implemented yet)
- Show FPS counter (for debugging)

#### 3.2 Inventory System
- Create inventory UI (grid-based)
- Implement drag-and-drop item movement
- Add item stacking (max 64 typically)
- Create inventory open/close toggle (E key)
- Implement hotbar number key selection (1-9)

#### 3.3 Item System
- Create item class (separate from blocks)
- Implement item rendering (2D icons vs 3D models)
- Add item metadata (durability, enchantments, etc.)
- Create item entity rendering (dropped items)

**⚠️ DESIGN DECISION:** Will items have 2D sprites or 3D models? Plan this early.

### Phase 4: Crafting System (Week 7)

#### 4.1 Crafting Interface
- Design and implement crafting table UI (3x3 grid)
- Add player crafting (2x2 grid)
- Create recipe matching system (shaped vs shapeless recipes)
- Implement result slot with crafting preview
- Add recipe book (optional - shows discovered recipes)

#### 4.2 Crafting Recipes
- Implement recipe data structure (JSON or Python dict)
- Add basic recipes:
  - Planks from logs
  - Sticks from planks
  - Tools (pickaxe, shovel, axe, sword, hoe)
  - Crafting table
  - Furnace
  - Torches
  - Chests (if time permits)
- Create recipe validation system
- Add recipe unlocking (optional)

#### 4.3 Tool System
- Implement tool durability
- Add tool break animation/sound
- Create tool effectiveness multipliers
- Implement tool repair (if time permits)

#### 4.4 Furnace System (Optional)
- Create furnace UI (input, fuel, output)
- Implement smelting recipes
- Add fuel burning mechanics
- Create progress indicators

**⚠️ COMPLEXITY:** Furnaces require persistent state and tick updates - may be complex

### Phase 5: Entities & Mobs (Weeks 8-9)

#### 5.1 Entity System
- Create base entity class:
  - Position, rotation, velocity
  - Bounding box
  - Health
  - Update and render methods
- Implement entity manager
- Add entity spawning system
- Create entity physics (gravity, collisions)
- Implement entity rendering (simple models or voxel-based)

#### 5.2 Item Entities
- Create dropped item entities
- Implement item pickup system
- Add item despawn timer (5 minutes)
- Create item merging (same items stack)

#### 5.3 Friendly Mobs
- Implement simple mob models:
  - Cow: Simple rectangular body + legs
  - Chicken: Smaller body + head
  - Sheep: Similar to cow (wool blocks)
- Add mob AI states:
  - Idle/standing
  - Walking/wandering
  - Fleeing (if hurt)
- Implement mob spawning in valid locations
- Add mob despawning (far from player)

#### 5.4 Mob Interactions
- Implement mob drops on death:
  - Cow: leather, raw beef
  - Chicken: feathers, raw chicken, eggs (randomly)
  - Sheep: wool (1-3), mutton
- Add breeding system (feed two animals)
- Implement baby mobs (scaled down, grow over time)
- Add mob sounds (procedural or simple)

**⚠️ PERFORMANCE:** Many mobs can impact performance - implement entity culling

### Phase 6: Polish & Core Features (Week 10)

#### 6.1 World Features
- Add trees (oak, birch)
- Implement simple cave generation (3D Perlin noise)
- Add ore generation (coal, iron, gold, diamond)
- Create water blocks (if time permits - complex rendering)
- Implement day/night cycle
- Add skybox rendering

#### 6.2 Lighting System
- Implement sunlight propagation (expensive!)
- Add block light sources (torches)
- Create light level calculations
- Update lighting when blocks change
- Add smooth lighting (ambient occlusion)

**⚠️ MAJOR COMPLEXITY:** Lighting is one of the hardest parts - may need simplified approach

#### 6.3 Audio System
- Implement procedural sound effects:
  - Block breaking (different per block type)
  - Footsteps
  - Item pickup
  - Mob sounds
  - Ambient sounds
- Add background music (optional)
- Create audio manager with volume control

#### 6.4 Particle Effects
- Create particle system
- Add particles for:
  - Block breaking
  - Item collection
  - Mob damage
  - Torch flames
- Implement particle physics (gravity, fading)

### Phase 7: Save/Load & Optimization (Week 11-12)

#### 7.1 Save System
- Design save file format (compressed binary or JSON)
- Implement chunk serialization (pickle or custom format)
- Save player data (position, inventory, health)
- Add world metadata (seed, time, etc.)
- Create save/load UI
- Implement auto-save system

**⚠️ CONSIDERATION:** Large worlds = large save files. Consider chunk-based saving.

#### 7.2 Performance Optimization
- Profile the game (cProfile, line_profiler)
- Optimize mesh generation (use Numba/Cython if needed)
- Implement frustum culling (don't render chunks outside view)
- Add chunk render distance setting
- Optimize entity updates (spatial partitioning)
- Implement VBO reuse (don't reallocate every frame)
- Add LOD system for distant chunks (optional)

#### 7.3 Bug Fixes & Testing
- Comprehensive play testing
- Fix collision bugs
- Fix rendering artifacts
- Test save/load thoroughly
- Test chunk loading/unloading edge cases
- Stress test with many mobs/items

### Phase 8: Final Polish (Week 13-14)

#### 8.1 UI/UX Improvements
- Add pause menu
- Create settings menu (render distance, FOV, controls)
- Implement death screen and respawn
- Add creative mode (optional)
- Create world selection screen
- Add ESC menu with save/quit

#### 8.2 Documentation
- Write README with setup instructions
- Create user guide
- Document code architecture
- Add inline code comments
- Create troubleshooting guide

#### 8.3 Final Testing
- Full play testing session
- Performance testing on different hardware
- Save/load stress testing
- Edge case testing

## Detailed Implementation Notes

### Critical Architecture Decisions

#### 1. Modern OpenGL vs Immediate Mode
**Your example code uses immediate mode (glBegin/glEnd) - this is deprecated and slow!**

❌ **DON'T USE:**
```python
glBegin(GL_QUADS)
glVertex3fv(vertex)
glEnd()
```

✅ **USE INSTEAD:** Modern OpenGL with VAO/VBO
```python
# Create vertex buffer once
vertices = np.array([...], dtype=np.float32)
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Create vertex array object
vao = glGenVertexArrays(1)
glBindVertexArray(vao)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

# Render (very fast)
glBindVertexArray(vao)
glDrawArrays(GL_TRIANGLES, 0, len(vertices))
```

#### 2. Chunk Mesh Generation Strategy

**Problem:** Rendering every block face individually is too slow (millions of faces)

**Solution:** Greedy Meshing Algorithm
- Merge adjacent blocks into larger quads
- Cull faces between solid blocks
- Generate one mesh per chunk
- Rebuild mesh only when chunk changes

**Example Approach:**
```python
class Chunk:
    def __init__(self, position):
        self.position = position
        self.blocks = np.zeros((16, 256, 16), dtype=np.uint8)  # Block IDs
        self.mesh = None  # VBO handle
        self.is_dirty = True  # Needs remeshing
    
    def build_mesh(self):
        """Generate optimized mesh for this chunk"""
        vertices = []
        
        for x in range(16):
            for y in range(256):
                for z in range(16):
                    block_id = self.blocks[x, y, z]
                    if block_id == 0:  # Air
                        continue
                    
                    # Check each face
                    for face in FACES:
                        nx, ny, nz = x + face.normal[0], y + face.normal[1], z + face.normal[2]
                        
                        # Check if neighbor is solid
                        if not self.is_block_solid(nx, ny, nz):
                            # Add this face to mesh
                            vertices.extend(face.get_vertices(x, y, z, block_id))
        
        # Upload to GPU
        self.upload_mesh(vertices)
        self.is_dirty = False
    
    def is_block_solid(self, x, y, z):
        """Check if block is solid, handling chunk boundaries"""
        if 0 <= x < 16 and 0 <= y < 256 and 0 <= z < 16:
            return self.blocks[x, y, z] != 0
        else:
            # Query neighbor chunk
            return self.world.is_block_solid_global(
                self.position[0] * 16 + x,
                y,
                self.position[1] * 16 + z
            )
```

#### 3. Raycasting for Block Selection

**Critical for mining/placing blocks**

```python
def raycast(start, direction, max_distance=10):
    """
    DDA raycasting algorithm to find block player is looking at
    Returns: (block_pos, face_normal) or (None, None)
    """
    # Current position
    x, y, z = start
    
    # Ray direction
    dx, dy, dz = direction
    
    # Step direction
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1
    
    # tMax: next t value when crossing voxel boundary
    t_max_x = intbound(x, dx) if dx != 0 else float('inf')
    t_max_y = intbound(y, dy) if dy != 0 else float('inf')
    t_max_z = intbound(z, dz) if dz != 0 else float('inf')
    
    # tDelta: t distance between voxel boundaries
    t_delta_x = step_x / dx if dx != 0 else float('inf')
    t_delta_y = step_y / dy if dy != 0 else float('inf')
    t_delta_z = step_z / dz if dz != 0 else float('inf')
    
    face = None
    
    while distance(start, (x, y, z)) < max_distance:
        # Check if current block is solid
        if world.get_block(int(x), int(y), int(z)) != 0:
            return ((int(x), int(y), int(z)), face)
        
        # Step to next voxel
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                x += step_x
                t_max_x += t_delta_x
                face = (-step_x, 0, 0)
            else:
                z += step_z
                t_max_z += t_delta_z
                face = (0, 0, -step_z)
        else:
            if t_max_y < t_max_z:
                y += step_y
                t_max_y += t_delta_y
                face = (0, -step_y, 0)
            else:
                z += step_z
                t_max_z += t_delta_z
                face = (0, 0, -step_z)
    
    return (None, None)

def intbound(s, ds):
    """Helper for DDA algorithm"""
    if ds < 0:
        return intbound(-s, -ds)
    else:
        s = s % 1
        return (1 - s) / ds
```

### 3D Rendering with Modern OpenGL

**Shader System:**
```python
# vertex.glsl
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in float lighting;

out vec2 fragTexCoord;
out float fragLighting;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    fragTexCoord = texCoord;
    fragLighting = lighting;
}

# fragment.glsl
#version 330 core
in vec2 fragTexCoord;
in float fragLighting;

out vec4 FragColor;

uniform sampler2D textureSampler;

void main() {
    vec4 texColor = texture(textureSampler, fragTexCoord);
    FragColor = texColor * fragLighting;
}
```

**Python Shader Loading:**
```python
class ShaderProgram:
    def __init__(self, vertex_path, fragment_path):
        with open(vertex_path, 'r') as f:
            vertex_source = f.read()
        with open(fragment_path, 'r') as f:
            fragment_source = f.read()
        
        # Compile shaders
        vertex_shader = self.compile_shader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(fragment_source, GL_FRAGMENT_SHADER)
        
        # Link program
        self.program = glCreateProgram()
        glAttachShader(self.program, vertex_shader)
        glAttachShader(self.program, fragment_shader)
        glLinkProgram(self.program)
        
        # Check for errors
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(self.program))
        
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
    
    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        
        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(shader))
        
        return shader
    
    def use(self):
        glUseProgram(self.program)
    
    def set_uniform_matrix(self, name, matrix):
        location = glGetUniformLocation(self.program, name)
        glUniformMatrix4fv(location, 1, GL_FALSE, matrix)
```

### Procedural Texture Generation
```python
# Example of procedural texture generation in block_textures.py
import numpy as np
from PIL import Image
import random

class TextureGenerator:
    def __init__(self, size=16):
        self.size = size
        self.textures = {}
    
    def generate_grass_texture(self):
        """Generate grass top texture"""
        texture = Image.new('RGB', (self.size, self.size))
        pixels = texture.load()
        
        # Base green color
        for x in range(self.size):
            for y in range(self.size):
                # Color variation
                r = random.randint(70, 100)
                g = random.randint(120, 180)
                b = random.randint(40, 80)
                
                # Add noise for texture detail
                if random.random() < 0.15:
                    r = min(255, r + random.randint(10, 30))
                    g = min(255, g + random.randint(10, 30))
                
                pixels[x, y] = (r, g, b)
        
        return texture
    
    def generate_dirt_texture(self):
        """Generate dirt texture"""
        texture = Image.new('RGB', (self.size, self.size))
        pixels = texture.load()
        
        for x in range(self.size):
            for y in range(self.size):
                # Brown color
                base = random.randint(80, 120)
                r = base + random.randint(-20, 20)
                g = int(base * 0.6) + random.randint(-15, 15)
                b = int(base * 0.4) + random.randint(-10, 10)
                
                # Clamp values
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                pixels[x, y] = (r, g, b)
        
        return texture
    
    def generate_stone_texture(self):
        """Generate stone texture"""
        texture = Image.new('RGB', (self.size, self.size))
        pixels = texture.load()
        
        for x in range(self.size):
            for y in range(self.size):
                # Gray with variation
                value = random.randint(80, 140)
                variation = random.randint(-20, 20)
                
                r = g = b = max(0, min(255, value + variation))
                
                pixels[x, y] = (r, g, b)
        
        return texture
    
    def generate_wood_texture(self):
        """Generate wood texture with grain"""
        texture = Image.new('RGB', (self.size, self.size))
        pixels = texture.load()
        
        for x in range(self.size):
            for y in range(self.size):
                # Vertical grain pattern
                grain = np.sin(x * 0.5) * 20
                
                # Brown color
                r = int(139 + grain + random.randint(-10, 10))
                g = int(90 + grain * 0.6 + random.randint(-10, 10))
                b = int(43 + grain * 0.3 + random.randint(-5, 5))
                
                # Clamp
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                pixels[x, y] = (r, g, b)
        
        return texture
    
    def create_texture_atlas(self):
        """Create a texture atlas from all textures"""
        # Generate all textures
        textures_dict = {
            'grass_top': self.generate_grass_texture(),
            'grass_side': self.generate_grass_side_texture(),
            'dirt': self.generate_dirt_texture(),
            'stone': self.generate_stone_texture(),
            'wood': self.generate_wood_texture(),
            'planks': self.generate_planks_texture(),
            'leaves': self.generate_leaves_texture(),
            'cobblestone': self.generate_cobblestone_texture(),
        }
        
        # Create atlas (8x8 grid for 64 textures)
        atlas_size = 8
        atlas = Image.new('RGB', (self.size * atlas_size, self.size * atlas_size))
        
        # Place textures in atlas
        for idx, (name, tex) in enumerate(textures_dict.items()):
            x = (idx % atlas_size) * self.size
            y = (idx // atlas_size) * self.size
            atlas.paste(tex, (x, y))
            
            # Store UV coordinates
            self.textures[name] = {
                'u': x / (self.size * atlas_size),
                'v': y / (self.size * atlas_size),
                'width': self.size / (self.size * atlas_size),
                'height': self.size / (self.size * atlas_size)
            }
        
        return atlas
    
    def generate_grass_side_texture(self):
        """Generate grass side (brown bottom, green top)"""
        texture = Image.new('RGB', (self.size, self.size))
        pixels = texture.load()
        
        for x in range(self.size):
            for y in range(self.size):
                if y < self.size * 0.7:  # Dirt portion
                    base = random.randint(80, 120)
                    r = base + random.randint(-20, 20)
                    g = int(base * 0.6) + random.randint(-15, 15)
                    b = int(base * 0.4) + random.randint(-10, 10)
                else:  # Grass portion
                    r = random.randint(70, 100)
                    g = random.randint(120, 180)
                    b = random.randint(40, 80)
                
                pixels[x, y] = (max(0, min(255, r)), 
                               max(0, min(255, g)), 
                               max(0, min(255, b)))
        
        return texture
    
    def generate_planks_texture(self):
        """Generate wooden planks texture"""
        texture = Image.new('RGB', (self.size, self.size))
        pixels = texture.load()
        
        for x in range(self.size):
            for y in range(self.size):
                # Horizontal planks
                plank_line = (y % 4 == 0)
                variation = -15 if plank_line else random.randint(-5, 5)
                
                r = max(0, min(255, 150 + variation))
                g = max(0, min(255, 100 + variation))
                b = max(0, min(255, 50 + variation))
                
                pixels[x, y] = (r, g, b)
        
        return texture
    
    def generate_leaves_texture(self):
        """Generate leaves texture"""
        texture = Image.new('RGBA', (self.size, self.size))
        pixels = texture.load()
        
        for x in range(self.size):
            for y in range(self.size):
                # Some transparent pixels
                if random.random() < 0.1:
                    pixels[x, y] = (0, 0, 0, 0)
                else:
                    r = random.randint(40, 80)
                    g = random.randint(100, 150)
                    b = random.randint(30, 70)
                    pixels[x, y] = (r, g, b, 255)
        
        return texture.convert('RGB')  # Convert back to RGB
    
    def generate_cobblestone_texture(self):
        """Generate cobblestone texture"""
        texture = Image.new('RGB', (self.size, self.size))
        pixels = texture.load()
        
        for x in range(self.size):
            for y in range(self.size):
                # Stone texture with darker lines
                is_line = (x % 4 == 0) or (y % 4 == 0)
                base = random.randint(70, 110) if not is_line else random.randint(50, 80)
                
                r = g = b = max(0, min(255, base + random.randint(-10, 10)))
                pixels[x, y] = (r, g, b)
        
        return texture
```

**⚠️ IMPORTANT:** Cache generated textures to disk to avoid regenerating every time!

### Entity AI System
```python
# Improved mob AI in ai.py
import math
import random
from enum import Enum

class AIState(Enum):
    IDLE = 0
    WANDERING = 1
    FLEEING = 2
    FOLLOWING = 3  # For breeding

class MobAI:
    def __init__(self, mob):
        self.mob = mob
        self.state = AIState.WANDERING
        self.state_timer = 0
        self.target_position = None
        self.wander_direction = self.get_random_direction()
        self.decision_cooldown = 0
    
    def get_random_direction(self):
        """Get random horizontal direction"""
        angle = random.uniform(0, 2 * math.pi)
        return [math.cos(angle), 0, math.sin(angle)]
    
    def update(self, world, player_pos, delta_time):
        """Update AI state and movement"""
        self.state_timer -= delta_time
        self.decision_cooldown -= delta_time
        
        # Calculate distance to player
        distance_to_player = self.calculate_distance(self.mob.position, player_pos)
        
        # State transitions
        if self.decision_cooldown <= 0:
            self.decision_cooldown = random.uniform(1.0, 3.0)
            
            if distance_to_player < 3.0:  # Player too close
                if self.state != AIState.FLEEING:
                    self.transition_to_fleeing(player_pos)
            elif self.state == AIState.FLEEING and distance_to_player > 8.0:
                self.transition_to_wandering()
            elif self.state == AIState.WANDERING and self.state_timer <= 0:
                # Random state change
                if random.random() < 0.3:
                    self.transition_to_idle()
                else:
                    self.wander_direction = self.get_random_direction()
                    self.state_timer = random.uniform(3.0, 8.0)
            elif self.state == AIState.IDLE and self.state_timer <= 0:
                self.transition_to_wandering()
        
        # Execute state behavior
        self.execute_state(world, player_pos, delta_time)
    
    def transition_to_idle(self):
        """Transition to idle state"""
        self.state = AIState.IDLE
        self.state_timer = random.uniform(2.0, 5.0)
        self.mob.velocity = [0, self.mob.velocity[1], 0]
    
    def transition_to_wandering(self):
        """Transition to wandering state"""
        self.state = AIState.WANDERING
        self.state_timer = random.uniform(3.0, 8.0)
        self.wander_direction = self.get_random_direction()
    
    def transition_to_fleeing(self, player_pos):
        """Transition to fleeing state"""
        self.state = AIState.FLEEING
        self.state_timer = random.uniform(4.0, 7.0)
        
        # Calculate flee direction (away from player)
        dx = self.mob.position[0] - player_pos[0]
        dz = self.mob.position[2] - player_pos[2]
        length = math.sqrt(dx*dx + dz*dz)
        
        if length > 0:
            self.wander_direction = [dx/length, 0, dz/length]
    
    def execute_state(self, world, player_pos, delta_time):
        """Execute current state behavior"""
        if self.state == AIState.IDLE:
            # Do nothing
            pass
        
        elif self.state == AIState.WANDERING:
            self.move_in_direction(world, self.wander_direction, 0.5, delta_time)
        
        elif self.state == AIState.FLEEING:
            self.move_in_direction(world, self.wander_direction, 1.5, delta_time)
        
        elif self.state == AIState.FOLLOWING:
            # Move toward target
            if self.target_position:
                direction = self.get_direction_to(self.target_position)
                self.move_in_direction(world, direction, 0.8, delta_time)
    
    def move_in_direction(self, world, direction, speed, delta_time):
        """Move mob in given direction"""
        # Apply movement
        self.mob.velocity[0] = direction[0] * speed
        self.mob.velocity[2] = direction[2] * speed
        
        # Check for obstacles (simple)
        new_x = self.mob.position[0] + self.mob.velocity[0] * delta_time
        new_z = self.mob.position[2] + self.mob.velocity[2] * delta_time
        
        # If blocked, change direction
        if self.is_blocked(world, new_x, self.mob.position[1], new_z):
            self.wander_direction = self.get_random_direction()
    
    def is_blocked(self, world, x, y, z):
        """Check if position is blocked"""
        # Check if block at position is solid
        block = world.get_block(int(x), int(y), int(z))
        return block != 0  # 0 = air
    
    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dz = pos1[2] - pos2[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_direction_to(self, target_pos):
        """Get normalized direction to target"""
        dx = target_pos[0] - self.mob.position[0]
        dz = target_pos[2] - self.mob.position[2]
        length = math.sqrt(dx*dx + dz*dz)
        
        if length > 0:
            return [dx/length, 0, dz/length]
        return [0, 0, 0]

# Specific mob implementations
class CowAI(MobAI):
    """Cow-specific AI behavior"""
    def __init__(self, mob):
        super().__init__(mob)
        self.milk_cooldown = 0
    
    def update(self, world, player_pos, delta_time):
        super().update(world, player_pos, delta_time)
        self.milk_cooldown = max(0, self.milk_cooldown - delta_time)

class ChickenAI(MobAI):
    """Chicken-specific AI behavior"""
    def __init__(self, mob):
        super().__init__(mob)
        self.egg_timer = random.uniform(30.0, 90.0)  # Lay egg every 30-90 seconds
    
    def update(self, world, player_pos, delta_time):
        super().update(world, player_pos, delta_time)
        
        # Egg laying
        self.egg_timer -= delta_time
        if self.egg_timer <= 0:
            self.lay_egg(world)
            self.egg_timer = random.uniform(30.0, 90.0)
    
    def lay_egg(self, world):
        """Spawn an egg item"""
        # Create egg item entity at chicken's position
        world.spawn_item('egg', self.mob.position)
```

**⚠️ NOTE:** AI needs pathfinding for complex navigation - for now, simple obstacle avoidance is sufficient.

### Crafting System
```python
# Improved crafting system in mechanics/crafting.py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class Recipe:
    """Represents a crafting recipe"""
    name: str
    ingredients: Dict[str, int]  # {item_name: count}
    result_item: str
    result_count: int
    shaped: bool = False  # Shaped vs shapeless recipe
    pattern: Optional[list] = None  # For shaped recipes

class CraftingSystem:
    def __init__(self):
        self.recipes = []
        self.setup_recipes()
    
    def setup_recipes(self):
        """Initialize all crafting recipes"""
        
        # Basic recipes
        self.add_recipe(Recipe(
            name="planks",
            ingredients={"log": 1},
            result_item="planks",
            result_count=4
        ))
        
        self.add_recipe(Recipe(
            name="sticks",
            ingredients={"planks": 2},
            result_item="stick",
            result_count=4,
            shaped=True,
            pattern=[
                ['planks'],
                ['planks']
            ]
        ))
        
        self.add_recipe(Recipe(
            name="crafting_table",
            ingredients={"planks": 4},
            result_item="crafting_table",
            result_count=1
        ))
        
        # Tool recipes
        self.add_recipe(Recipe(
            name="wooden_pickaxe",
            ingredients={"planks": 3, "stick": 2},
            result_item="wooden_pickaxe",
            result_count=1,
            shaped=True,
            pattern=[
                ['planks', 'planks', 'planks'],
                ['', 'stick', ''],
                ['', 'stick', '']
            ]
        ))
        
        self.add_recipe(Recipe(
            name="wooden_axe",
            ingredients={"planks": 3, "stick": 2},
            result_item="wooden_axe",
            result_count=1,
            shaped=True,
            pattern=[
                ['planks', 'planks', ''],
                ['planks', 'stick', ''],
                ['', 'stick', '']
            ]
        ))
        
        self.add_recipe(Recipe(
            name="wooden_shovel",
            ingredients={"planks": 1, "stick": 2},
            result_item="wooden_shovel",
            result_count=1,
            shaped=True,
            pattern=[
                ['planks'],
                ['stick'],
                ['stick']
            ]
        ))
        
        # Stone tool variants
        for tool in ['pickaxe', 'axe', 'shovel', 'sword']:
            count = 3 if tool != 'shovel' else 1
            self.add_stone_tool_recipe(tool, count)
        
        # Special items
        self.add_recipe(Recipe(
            name="torch",
            ingredients={"coal": 1, "stick": 1},
            result_item="torch",
            result_count=4,
            shaped=True,
            pattern=[
                ['coal'],
                ['stick']
            ]
        ))
        
        self.add_recipe(Recipe(
            name="furnace",
            ingredients={"cobblestone": 8},
            result_item="furnace",
            result_count=1
        ))
    
    def add_stone_tool_recipe(self, tool_name, material_count):
        """Helper to add stone tool recipes"""
        patterns = {
            'pickaxe': [
                ['cobblestone', 'cobblestone', 'cobblestone'],
                ['', 'stick', ''],
                ['', 'stick', '']
            ],
            'axe': [
                ['cobblestone', 'cobblestone', ''],
                ['cobblestone', 'stick', ''],
                ['', 'stick', '']
            ],
            'shovel': [
                ['cobblestone'],
                ['stick'],
                ['stick']
            ],
            'sword': [
                ['cobblestone'],
                ['cobblestone'],
                ['stick']
            ]
        }
        
        self.add_recipe(Recipe(
            name=f"stone_{tool_name}",
            ingredients={"cobblestone": material_count, "stick": 2 if tool_name != 'sword' else 1},
            result_item=f"stone_{tool_name}",
            result_count=1,
            shaped=True,
            pattern=patterns[tool_name]
        ))
    
    def add_recipe(self, recipe):
        """Add a recipe to the system"""
        self.recipes.append(recipe)
    
    def can_craft(self, recipe_name, inventory):
        """Check if player has ingredients for recipe"""
        recipe = self.get_recipe(recipe_name)
        if not recipe:
            return False
        
        for item, count in recipe.ingredients.items():
            if inventory.get_item_count(item) < count:
                return False
        
        return True
    
    def craft(self, recipe_name, inventory):
        """Attempt to craft an item"""
        recipe = self.get_recipe(recipe_name)
        if not recipe or not self.can_craft(recipe_name, inventory):
            return False
        
        # Remove ingredients
        for item, count in recipe.ingredients.items():
            inventory.remove_item(item, count)
        
        # Add result
        inventory.add_item(recipe.result_item, recipe.result_count)
        
        return True
    
    def get_recipe(self, recipe_name):
        """Get recipe by name"""
        for recipe in self.recipes:
            if recipe.name == recipe_name:
                return recipe
        return None
    
    def get_available_recipes(self, inventory):
        """Get list of recipes player can currently craft"""
        available = []
        for recipe in self.recipes:
            if self.can_craft(recipe.name, inventory):
                available.append(recipe)
        return available
    
    def match_crafting_grid(self, grid, grid_size):
        """
        Match a crafting grid pattern to recipes
        grid: 2D array of item names ('' for empty)
        grid_size: 2 for 2x2, 3 for 3x3
        """
        # Try shaped recipes first
        for recipe in self.recipes:
            if recipe.shaped and recipe.pattern:
                if self.matches_pattern(grid, recipe.pattern, grid_size):
                    return recipe
        
        # Try shapeless recipes
        grid_items = {}
        for row in grid:
            for item in row:
                if item:
                    grid_items[item] = grid_items.get(item, 0) + 1
        
        for recipe in self.recipes:
            if not recipe.shaped and recipe.ingredients == grid_items:
                return recipe
        
        return None
    
    def matches_pattern(self, grid, pattern, grid_size):
        """Check if grid matches recipe pattern"""
        pattern_height = len(pattern)
        pattern_width = len(pattern[0]) if pattern else 0
        
        # Try all possible positions in grid
        for start_row in range(grid_size - pattern_height + 1):
            for start_col in range(grid_size - pattern_width + 1):
                if self.check_pattern_at_position(grid, pattern, start_row, start_col, grid_size):
                    return True
        
        return False
    
    def check_pattern_at_position(self, grid, pattern, start_row, start_col, grid_size):
        """Check if pattern matches at specific position"""
        # Check if rest of grid is empty
        for row in range(grid_size):
            for col in range(grid_size):
                in_pattern = (start_row <= row < start_row + len(pattern) and
                            start_col <= col < start_col + len(pattern[0]))
                
                if in_pattern:
                    pattern_row = row - start_row
                    pattern_col = col - start_col
                    expected = pattern[pattern_row][pattern_col] if pattern_col < len(pattern[pattern_row]) else ''
                    
                    if grid[row][col] != expected:
                        return False
                else:
                    if grid[row][col] != '':
                        return False
        
        return True

# Inventory class
class Inventory:
    def __init__(self, size=36):
        self.size = size
        self.slots = [None] * size  # Each slot: {'item': str, 'count': int} or None
        self.max_stack_size = 64
    
    def add_item(self, item_name, count):
        """Add items to inventory, returns remaining count if inventory full"""
        remaining = count
        
        # Try to add to existing stacks
        for i, slot in enumerate(self.slots):
            if slot and slot['item'] == item_name:
                can_add = min(remaining, self.max_stack_size - slot['count'])
                slot['count'] += can_add
                remaining -= can_add
                
                if remaining == 0:
                    return 0
        
        # Create new stacks
        for i, slot in enumerate(self.slots):
            if slot is None:
                can_add = min(remaining, self.max_stack_size)
                self.slots[i] = {'item': item_name, 'count': can_add}
                remaining -= can_add
                
                if remaining == 0:
                    return 0
        
        return remaining  # Couldn't fit all items
    
    def remove_item(self, item_name, count):
        """Remove items from inventory, returns True if successful"""
        if self.get_item_count(item_name) < count:
            return False
        
        remaining = count
        for slot in self.slots:
            if slot and slot['item'] == item_name:
                can_remove = min(remaining, slot['count'])
                slot['count'] -= can_remove
                remaining -= can_remove
                
                if slot['count'] == 0:
                    self.slots[self.slots.index(slot)] = None
                
                if remaining == 0:
                    return True
        
        return False
    
    def get_item_count(self, item_name):
        """Get total count of an item in inventory"""
        total = 0
        for slot in self.slots:
            if slot and slot['item'] == item_name:
                total += slot['count']
        return total
    
    def has_space(self):
        """Check if inventory has any empty slots"""
        return None in self.slots
```

**⚠️ IMPORTANT:** Crafting grid UI needs to call `match_crafting_grid()` whenever grid changes to show result preview!

## Major Pitfalls & Solutions

### 1. **Performance Bottlenecks**

#### Problem: Python is slow for real-time 3D rendering
**Solutions:**
- Use NumPy for all array operations
- Consider Numba JIT compilation for hot paths (mesh generation, physics)
- Batch render calls (one draw call per chunk, not per block)
- Implement aggressive culling (frustum, occlusion, face culling)
- Limit chunk updates per frame (spread over multiple frames)

#### Problem: Mesh generation is expensive
**Solutions:**
- Generate meshes on separate thread (Python threading for I/O, multiprocessing for CPU)
- Implement mesh caching - only rebuild when chunk changes
- Use greedy meshing algorithm to reduce vertex count
- Prioritize chunks closer to player

### 2. **Memory Management**

#### Problem: Large worlds consume too much memory
**Solutions:**
- Unload distant chunks from memory
- Compress chunk data when not active
- Use smaller data types (uint8 for block IDs instead of int)
- Implement chunk serialization to disk

#### Problem: Too many VBOs
**Solutions:**
- Reuse VBO objects when chunks are unloaded
- Batch multiple chunks into single VBO (instancing)
- Implement VBO pool

### 3. **Rendering Issues**

#### Problem: Z-fighting (flickering between overlapping faces)
**Solutions:**
- Ensure faces don't overlap exactly
- Use depth buffer with sufficient precision
- Offset transparent blocks slightly

#### Problem: Transparent blocks (glass, water, leaves)
**Solutions:**
- Render transparent blocks in separate pass, back-to-front
- Sort transparent blocks by distance from camera
- Use alpha testing for leaves (binary transparent/opaque)

#### Problem: Lighting looks bad
**Solutions:**
- Implement smooth lighting (average light from adjacent blocks)
- Add ambient occlusion for corners
- Use simple directional light + ambient light (avoid complex lighting initially)

### 4. **Chunk Loading Issues**

#### Problem: Chunks pop in suddenly (jarring)
**Solutions:**
- Generate chunks in concentric circles around player
- Fade in new chunks (optional)
- Generate distant chunks with lower detail first

#### Problem: Chunks don't connect seamlessly
**Solutions:**
- Ensure chunks share vertex data at boundaries
- Rebuild neighbor chunks when edge blocks change
- Handle chunk coordinate transitions carefully

#### Problem: Chunk data at boundaries
**Solutions:**
- When generating mesh, query neighbor chunks for adjacent blocks
- Cache neighbor chunk references
- Handle null neighbor chunks (world edge)

### 5. **Physics & Collision**

#### Problem: Player falls through floor
**Solutions:**
- Use continuous collision detection (not just discrete)
- Check collision multiple times per frame if moving fast
- Separate collision checks for each axis (X, Y, Z)

#### Problem: Player gets stuck in walls
**Solutions:**
- Implement proper AABB collision response
- Add small epsilon offset to prevent floating point errors
- Test corner cases (moving into corners, slopes)

#### Problem: Entity collision with chunks
**Solutions:**
- Only check collisions with nearby blocks (3x3x3 around entity)
- Use entity bounding box, not just point
- Handle Y-axis (gravity) separately from XZ movement

### 6. **State Management**

#### Problem: Game state becomes inconsistent
**Solutions:**
- Single source of truth for world state
- Clear ownership of data (who modifies what)
- Use event system for inter-component communication
- Careful with threading - protect shared state

#### Problem: UI state vs game state
**Solutions:**
- Separate UI state from game logic
- UI reads from game state, sends commands to modify it
- Don't store gameplay data in UI components

### 7. **Input Handling**

#### Problem: Mouse capture issues
**Solutions:**
- Handle ESC key to release mouse
- Grab mouse only when window has focus
- Show cursor when in inventory/menu

#### Problem: Input lag or missed inputs
**Solutions:**
- Process input events every frame
- Use event-driven input for key presses
- Use polling for continuous input (movement)

### 8. **Save/Load System**

#### Problem: Save files are huge
**Solutions:**
- Only save modified chunks (not default generated terrain)
- Use compression (gzip, lz4)
- Save chunk data in binary format, not text

#### Problem: Save corruption
**Solutions:**
- Write to temporary file first, then rename (atomic operation)
- Include version number in save format
- Validate data when loading
- Keep backup of previous save

#### Problem: Loading takes too long
**Solutions:**
- Load chunks asynchronously
- Show loading screen
- Load chunks near player first
- Cache generated chunks to disk

### 9. **Procedural Generation**

#### Problem: Terrain looks repetitive
**Solutions:**
- Use multiple octaves of noise (fractal noise)
- Combine different noise functions
- Add biomes with different parameters
- Place features (trees, rocks) using different seed

#### Problem: Biome transitions are harsh
**Solutions:**
- Use noise for biome blending
- Interpolate between biome parameters
- Create transition biomes

#### Problem: Structures (trees) spawn incorrectly
**Solutions:**
- Check if space is available before placing
- Place structures after terrain generation
- Use structure templates or procedural generation

### 10. **Lighting System**

#### Problem: Lighting is too slow
**Solutions:**
- Use flood-fill algorithm for light propagation
- Limit light propagation per frame (spread over time)
- Simplify to just sunlight + basic block light (no colored light)
- Consider baked lighting (pre-calculate)

#### Problem: Light updates when breaking blocks
**Solutions:**
- Queue light updates, process gradually
- Only update affected area, not entire chunk
- Use light decrease queue and light increase queue

### 11. **OpenGL Context Issues**

#### Problem: Black screen or crashes
**Solutions:**
- Check OpenGL version supported by system
- Enable debug output in development
- Validate shader compilation
- Check for GL errors after each call (in development)

#### Problem: Texture atlas issues
**Solutions:**
- Use GL_NEAREST for block textures (no blur between tiles)
- Add padding between atlas tiles (prevent bleeding)
- Generate mipmaps correctly or disable them

### 12. **Modular Architecture**

#### Problem: Circular imports
**Solutions:**
- Design clear dependency hierarchy
- Use dependency injection
- Move shared code to utils
- Import inside functions if needed (not ideal)

#### Problem: Tight coupling between modules
**Solutions:**
- Define clear interfaces
- Use event system for loose coupling
- Dependency inversion (depend on abstractions)

### 13. **Development Workflow**

#### Problem: Hard to test specific features
**Solutions:**
- Create debug commands (teleport, spawn items, etc.)
- Add debug overlay (show chunk borders, collision boxes)
- Implement creative mode (fly, instant break, infinite items)
- Use seed for reproducible worlds

#### Problem: Slow iteration time
**Solutions:**
- Hot reload for some components (not everything)
- Save development worlds for quick testing
- Profile regularly to find bottlenecks early

### 14. **Cross-Platform Issues**

#### Problem: Different OpenGL versions on different platforms
**Solutions:**
- Test on multiple platforms early
- Use OpenGL 3.3 core profile (widely supported)
- Check for extensions before using them
- Have fallback rendering paths

## Testing Strategy

### 1. Unit Testing
Test individual components in isolation:

**Test Coverage:**
- Block system (block properties, registry)
- Crafting recipes (matching, validation)
- Inventory operations (add, remove, stacking)
- Noise generation (deterministic output)
- Collision detection (AABB math)
- Raycasting (hit detection)

**Example Test:**
```python
# tests/test_crafting.py
import pytest
from mechanics.crafting import CraftingSystem, Inventory

def test_craft_wooden_pickaxe():
    crafting = CraftingSystem()
    inventory = Inventory()
    
    # Add required items
    inventory.add_item('planks', 3)
    inventory.add_item('stick', 2)
    
    # Craft pickaxe
    result = crafting.craft('wooden_pickaxe', inventory)
    assert result == True
    assert inventory.get_item_count('wooden_pickaxe') == 1
    assert inventory.get_item_count('planks') == 0
    assert inventory.get_item_count('stick') == 0

def test_craft_insufficient_materials():
    crafting = CraftingSystem()
    inventory = Inventory()
    
    inventory.add_item('planks', 2)  # Need 3
    inventory.add_item('stick', 2)
    
    result = crafting.craft('wooden_pickaxe', inventory)
    assert result == False
    assert inventory.get_item_count('planks') == 2  # Unchanged
```

### 2. Integration Testing
Test component interactions:

**Test Scenarios:**
- World generation → chunk loading → mesh generation
- Player input → raycasting → block breaking → item drop
- Item pickup → inventory add → crafting → tool usage
- Mob AI → physics → collision → rendering
- Chunk changes → neighbor chunk updates → lighting updates

**Example Test:**
```python
# tests/test_world.py
def test_chunk_generation_and_loading():
    world = World(seed=12345)
    
    # Generate chunk at origin
    chunk = world.get_or_generate_chunk(0, 0)
    
    assert chunk is not None
    assert chunk.blocks.shape == (16, 256, 16)
    assert chunk.mesh is not None
    
    # Check neighbor chunks are linked
    neighbor = world.get_or_generate_chunk(1, 0)
    assert neighbor is not None
    
    # Modify block at chunk boundary
    world.set_block(15, 64, 0, 1)  # Set block at edge
    
    # Both chunks should be marked dirty
    assert chunk.is_dirty == True
    assert neighbor.is_dirty == True
```

### 3. Performance Testing
Measure and benchmark critical paths:

**Metrics to Track:**
- FPS (frames per second) - target: 60 FPS
- Mesh generation time - target: <16ms per chunk
- Chunk loading time - target: <100ms
- Memory usage - target: <2GB for large world
- Save/load time - target: <5s

**Benchmarking:**
```python
# tests/test_performance.py
import time
import cProfile

def benchmark_mesh_generation():
    chunk = Chunk((0, 0))
    # Fill with random blocks
    chunk.blocks = np.random.randint(0, 5, (16, 256, 16), dtype=np.uint8)
    
    start = time.perf_counter()
    chunk.build_mesh()
    duration = time.perf_counter() - start
    
    print(f"Mesh generation: {duration*1000:.2f}ms")
    assert duration < 0.016  # Must complete in one frame at 60fps

def profile_render_loop():
    # Profile entire frame
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(100):
        game.update()
        game.render()
    
    profiler.disable()
    profiler.print_stats(sort='cumtime')
```

### 4. Visual/Manual Testing
Regular playtesting sessions:

**Test Checklist:**
- [ ] Camera movement feels smooth
- [ ] Block breaking visual feedback works
- [ ] Block placement feels responsive
- [ ] Inventory UI is usable
- [ ] Crafting interface is intuitive
- [ ] Collision detection prevents going through walls
- [ ] Jumping and gravity feel natural
- [ ] Mobs move believably
- [ ] No visual artifacts (z-fighting, texture bleeding)
- [ ] No performance issues (stuttering, frame drops)
- [ ] Chunks load smoothly without pop-in
- [ ] Lighting looks acceptable
- [ ] Audio feedback is appropriate

### 5. Stress Testing
Test edge cases and limits:

**Scenarios:**
- Spawn 100+ mobs - check FPS impact
- Generate 1000+ chunks - check memory usage
- Fill inventory completely - check UI handles it
- Break/place blocks rapidly - check stability
- Travel at high speed - check chunk loading
- Save/load large world - check corruption
- Run for extended period - check memory leaks

### 6. Regression Testing
Prevent reintroducing bugs:

**Process:**
- Maintain list of fixed bugs
- Create test for each bug fix
- Run full test suite before each commit
- Use CI/CD if possible (GitHub Actions)

### 7. Bug Tracking & Prioritization

**Priority Levels:**
1. **Critical** - Crashes, data loss, unplayable
2. **High** - Major features broken, significant performance issues
3. **Medium** - Minor features broken, small visual issues
4. **Low** - Polish issues, nice-to-have improvements

**Bug Report Template:**
```
Title: [Brief description]
Priority: [Critical/High/Medium/Low]
Steps to reproduce:
1. 
2. 
3. 
Expected behavior:
Actual behavior:
System info: [OS, Python version, GPU]
```

## Development Best Practices

### 1. Version Control
- Commit frequently with clear messages
- Use branches for major features
- Tag releases/milestones
- Don't commit generated files (textures, saves, __pycache__)

### 2. Code Organization
- Follow PEP 8 style guide
- Use type hints for function signatures
- Document complex algorithms
- Keep functions small and focused
- Avoid globals - pass dependencies explicitly

### 3. Configuration Management
```python
# config.py - Centralize all magic numbers
class Config:
    # Display
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    FOV = 70
    FPS_TARGET = 60
    
    # World
    CHUNK_SIZE = 16
    CHUNK_HEIGHT = 256
    RENDER_DISTANCE = 8  # chunks
    WORLD_SEED = None  # None = random
    
    # Physics
    GRAVITY = -32.0  # blocks per second squared
    JUMP_VELOCITY = 10.0
    PLAYER_SPEED = 4.3  # blocks per second
    PLAYER_HEIGHT = 1.8
    PLAYER_WIDTH = 0.6
    
    # Performance
    MAX_CHUNK_UPDATES_PER_FRAME = 4
    MESH_GENERATION_THREADS = 4
    
    # Gameplay
    REACH_DISTANCE = 5.0
    BLOCK_BREAK_SPEED = 1.0
    MAX_INVENTORY_SIZE = 36
    HOTBAR_SIZE = 9
```

### 4. Logging
```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Usage
logger.info("Starting game")
logger.warning("Chunk generation taking longer than expected")
logger.error("Failed to load shader: %s", shader_path)
logger.debug("Player position: %s", player.position)
```

### 5. Error Handling
```python
# Graceful degradation
try:
    shader = load_shader("vertex.glsl", "fragment.glsl")
except ShaderCompilationError as e:
    logger.error("Shader compilation failed: %s", e)
    # Fall back to simple rendering
    shader = get_fallback_shader()

# Resource cleanup
try:
    game.run()
finally:
    game.cleanup()  # Release OpenGL resources
    pygame.quit()
```

### 6. Performance Profiling
```python
# Use context manager for timing
class Timer:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        duration = time.perf_counter() - self.start
        if duration > 0.016:  # Longer than one frame
            logger.warning(f"{self.name} took {duration*1000:.2f}ms")

# Usage
with Timer("Chunk meshing"):
    chunk.build_mesh()
```

### 7. Debug Tools
Implement debug commands and overlays:
```python
class DebugOverlay:
    def render(self, game):
        info = [
            f"FPS: {game.fps:.1f}",
            f"Position: {game.player.position}",
            f"Chunk: {game.world.get_chunk_pos(game.player.position)}",
            f"Loaded chunks: {len(game.world.chunks)}",
            f"Entities: {len(game.entities)}",
            f"Memory: {get_memory_usage():.1f}MB"
        ]
        render_text_lines(info, 10, 10)

class DebugCommands:
    @staticmethod
    def teleport(x, y, z):
        player.position = [x, y, z]
    
    @staticmethod
    def give_item(item_name, count=1):
        inventory.add_item(item_name, count)
    
    @staticmethod
    def toggle_fly():
        player.flying = not player.flying
    
    @staticmethod
    def set_time(time):
        world.time = time
```

## Project Timeline & Milestones

### Week 1: Foundation
**Milestone:** Basic rendering and camera movement
- Setup complete, window opens
- Camera can move and look around
- Single colored cube renders

### Week 2-3: Core Engine
**Milestone:** Basic voxel world you can walk in
- Terrain generates from noise
- Chunks load/unload
- Player has collision and gravity
- Can walk around in world

### Week 4-5: Block Interaction
**Milestone:** Can break and place blocks
- Raycast block selection works
- Block breaking animation
- Blocks drop as items
- Can place blocks from hotbar

### Week 6: UI & Inventory
**Milestone:** Functional inventory system
- Inventory UI opens and closes
- Can move items between slots
- Hotbar shows items
- Item counts display

### Week 7: Crafting
**Milestone:** Basic crafting works
- Can craft wooden tools
- Crafting table works
- Recipe matching functions
- Tool durability works

### Week 8-9: Entities
**Milestone:** Mobs spawn and move around
- Cows and chickens spawn
- Mobs wander realistically
- Can be killed for drops
- Breeding works

### Week 10: World Features
**Milestone:** World looks interesting
- Trees generate
- Ores spawn underground
- Day/night cycle works
- Basic lighting implemented

### Week 11-12: Save/Load & Optimization
**Milestone:** Stable, performant game
- Worlds save and load
- Runs at 60 FPS with 8 chunk render distance
- No major bugs
- Memory usage stable

### Week 13-14: Polish
**Milestone:** Complete game ready to play
- Menus work (pause, settings, world selection)
- Audio implemented
- Documentation complete
- All major features tested

## Risk Assessment

### High Risk Items
1. **Lighting system** - Very complex, may need simplification
2. **Performance** - Python may be too slow, might need Cython
3. **Transparent blocks** - Rendering order issues
4. **Threading** - Python GIL limits parallelism
5. **Memory usage** - Large worlds may use too much RAM

### Mitigation Strategies
- Start with simplest possible implementation
- Profile early and often
- Have fallback approaches ready
- Be willing to cut features if needed
- Focus on core gameplay first

### Contingency Plans
- **If performance too slow:** Reduce render distance, use Numba/Cython
- **If lighting too complex:** Use simple ambient + directional light only
- **If entities lag:** Limit entity count, simplify AI
- **If memory issues:** More aggressive chunk unloading
- **If time runs out:** Cut features in order: breeding → audio → lighting → furnace
## Future Enhancements (Post-Initial Release)

### Phase 2 Features
- **More biomes**: Desert, snow, jungle, ocean
- **Weather system**: Rain, snow, clouds
- **Better caves**: Cave systems, ravines, mineshafts
- **More mobs**: Hostile mobs (zombies, skeletons), more animals
- **Enchanting system**: Tool enhancements
- **Armor system**: Protection, visual equipment
- **Hunger system**: Food requirement, health regeneration
- **Beds**: Sleep system, set spawn point

### Advanced Features
- **Multiplayer support** (very complex - requires networking)
- **Redstone-like mechanics**: Logic gates, pistons, automation
- **Minecarts and rails**: Transportation system
- **Villages**: Procedural structures, NPCs
- **The Nether**: Alternate dimension
- **Boss mobs**: Ender dragon equivalent
- **Command system**: Admin commands

### Technical Improvements
- **Better performance**: Rewrite hot paths in Cython/C++
- **Mod support**: Plugin API for extensions
- **Shader customization**: Shader packs
- **Resource packs**: Custom textures and models
- **Advanced lighting**: Ray traced lighting, shadows
- **Better water**: Fluid dynamics, waves
- **LOD system**: Distant chunk simplification

## Key Takeaways & Success Criteria

### Must-Have Features (MVP)
✅ Functional voxel world with terrain generation  
✅ Player movement, collision, and physics  
✅ Block breaking and placement  
✅ Inventory system  
✅ Basic crafting (wooden/stone tools)  
✅ At least 2 passive mobs (cow, chicken)  
✅ Save/load system  
✅ Stable 60 FPS on moderate hardware  

### Nice-to-Have Features
- Furnace and smelting
- Breeding system
- Day/night cycle
- Basic lighting
- Trees and ores
- Audio system
- Creative mode

### Can Be Cut If Needed
- Particle effects
- Smooth lighting
- Advanced mob AI
- Recipe book
- Multiple biomes
- Complex structures

## Final Recommendations

### Critical Success Factors
1. **Start with modern OpenGL** - Don't use immediate mode
2. **Profile early** - Find performance issues before they become problems
3. **Keep it simple** - Don't over-engineer, especially initially
4. **Test frequently** - Small bugs become big problems
5. **Manage scope** - Be willing to cut features

### Common Mistakes to Avoid
❌ Using immediate mode OpenGL (glBegin/glEnd)  
❌ Not implementing face culling (rendering invisible faces)  
❌ Regenerating chunk meshes every frame  
❌ Not using VBOs/VAOs properly  
❌ Implementing complex features before basics work  
❌ Not profiling until the end  
❌ Trying to match Minecraft exactly  

### Development Priorities
1. **Get something rendering** (even just a cube)
2. **Camera and input** working smoothly
3. **Chunk system** with basic terrain
4. **Player physics** and collision
5. **Block interaction** (break/place)
6. **Inventory and crafting**
7. **Everything else**

### When to Pivot
Consider changing approach if:
- Can't maintain 30+ FPS with 4 chunk render distance
- Memory usage exceeds 4GB with small world
- Chunk generation takes >1 second per chunk
- Physics feels wrong after significant tuning
- Save files consistently corrupt

### Tools & Resources
- **OpenGL Tutorial**: learnopengl.com
- **PyOpenGL Guide**: Use VAOs/VBOs, not immediate mode
- **Profiling**: cProfile, line_profiler, memory_profiler
- **Debugging**: OpenGL debug output, RenderDoc for GL debugging
- **Inspiration**: Minecraft modding tutorials (Java → Python concepts)

---

## Summary of Plan Improvements

This plan has been significantly enhanced with the following key improvements:

### 1. Added Phase 0 - Project Setup & Foundation
Critical foundation work including environment setup, testing framework, and core architecture.

### 2. Restructured Folder Structure
Added missing modules: chunk.py, mesh_builder.py, input_handler.py, player.py, hotbar.py, raycast.py, shaders/, tests/, audio/

### 3. Modern OpenGL Implementation
**CRITICAL FIX:** Replaced deprecated immediate mode with modern VAO/VBO approach and shader system.

### 4. Comprehensive Implementation Examples
- Chunk mesh generation with face culling
- DDA raycasting for block selection
- Complete crafting system
- Full inventory implementation
- State-based AI system
- Texture atlas generation

### 5. Major Pitfalls & Solutions
14 major categories covering performance, rendering, physics, state management, and more.

### 6. Realistic Timeline
Extended to 14 weeks with detailed weekly milestones.

### 7. Testing Strategy
Unit tests, integration tests, performance benchmarks, visual testing, stress testing.

### 8. Development Best Practices
Configuration management, logging, error handling, profiling tools, debug commands.

### 9. Risk Assessment
High-risk items identified with mitigation strategies and contingency plans.

### 10. Clear Success Criteria
Must-have features, nice-to-have features, and cut priorities.

---

## Conclusion

The plan is now production-ready with realistic scope, modern implementation, comprehensive error prevention, and clear success criteria. **Start with Phase 0 and work systematically, testing each component before moving forward!**

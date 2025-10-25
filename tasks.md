# Pycraft - Project Task Checklist

This file is a checklist-style breakdown of the project plan (derived from `plan.md` and `QUICK_REFERENCE.md`). Use it to track progress by checking boxes as you complete subtasks.

Guiding constraints (from plan & quick reference):
- Use modern OpenGL (VAO/VBO/shaders). No immediate mode.
- Chunk size: 16x256x16 (configurable in `config.py`), use NumPy (`uint8`) for block storage.
- Mesh per chunk, greedy meshing, cache meshes, rebuild only when dirty.
- Use DDA raycasting for block selection.
- Profile early; targets: 60 FPS, <16ms mesh generation, <100ms chunk load.
- Save safely (write temp → rename). Use compressed chunk-based saves.

---

## How to use this file
- Check a box when a subtask is done. Keep high-level tasks unchecked until their subtasks are done.
- Add new subtasks as needed.
- Link commits or issue numbers next to completed items if helpful.

---

## Phase 0 — Project Setup & Foundation

### 0.1 Environment & repo
- [x] 0.1.1 Create virtual environment and activation script
  - [x] Create venv and document activation (Windows WSL / Windows instructions)
  - [x] Add `requirements.txt` and `pyproject.toml` or similar
  - [x] Add `.gitignore` (exclude saves/, resources/textures/, __pycache__)
  - Acceptance: `venv` works and `pip install -r requirements.txt` installs without error on dev machine.

- [x] 0.1.2 Setup initial git repo and CI scaffold (optional)
  - [x] Initialize git, add initial commit
 

### 0.2 Core architecture & config
- [x] 0.2.1 Create `config.py` with central constants
  - [x] Include CHUNK_SIZE, CHUNK_HEIGHT, RENDER_DISTANCE, FOV, FPS_TARGET, REACH_DISTANCE
  - [x] Document default values and how to override them
  - Acceptance: All modules import settings from `config.py`.

- [x] 0.2.2 Logging & error handling
  - [x] Implement `logging` config (file + console)
  - [x] Add a `logger` wrapper used by major modules
  - Acceptance: `logger.info(...)` outputs to console and `game.log`.

- [x] 0.2.3 Shader loader & asset layout
  - [x] Create `shaders/` folder and minimal `vertex.glsl`/`fragment.glsl`
  - [x] Implement `engine/renderer.py` shader compile/link utilities (with error reporting)
  - Acceptance: Simple shader compiles and `Renderer` reports shader link errors.

### 0.3 Testing scaffold
- [x] 0.3.1 Choose test framework (pytest recommended)
  - [x] Create `tests/` folder and example tests (utils, crafting)
  - Acceptance: `pytest` runs and example tests pass.

- [x] 0.3.2 Add basic performance measurement helpers
  - [x] `Timer` context manager and quick benchmark script for mesh generation
  - Acceptance: Timer logs long-running sections (>16 ms)

---

## Phase 1 — Core Engine (Rendering, Chunks, World)

### 1.1 Basic 3D rendering & camera
- [x] 1.1.1 Create Pygame window with OpenGL context (OpenGL 3.3 core)
  - [x] Ensure proper GL context flags for modern profile
  - [x] Add robust GL version check and fallback/logging
  - Acceptance: Program opens window and prints GL version >= 3.3 or logs a clear error.

- [x] 1.1.2 Implement `engine/shader.py` helper class
  - [x] Read shader sources, compile, link, and validate uniforms
  - [x] Provide `use()`, `set_uniform_matrix()` helpers
  - Acceptance: Vertex and fragment shader compile successfully for the simple textured cube example.

- [x] 1.1.3 Implement `engine/renderer.py` minimal rendering pipeline
  - [x] VAO/VBO creation utilities
  - [x] Texture upload + atlas support (GL_NEAREST, padding)
  - [x] Mesh draw function (bind VAO, glDrawElements / glDrawArrays)
  - Acceptance: A single colored cube renders using VBO/VAO and a shader program.

- [x] 1.1.4 Create `engine/camera.py` with FPS-style control
  - [x] Mouse look, WASD movement, ESC to release mouse
  - [x] Provide view and projection matrices
  - Acceptance: Camera can move and look around; crosshair+movement test OK.

### 1.2 Chunk data & mesh building
- [ ] 1.2.1 Implement `engine/chunk.py` data structure
  - [ ] Use `np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)` for block ids
  - [ ] Metadata: position (chunk coords), is_dirty flag, mesh handle
  - Acceptance: Chunk instance initializes and can be serialized (test chunk unit test)

- [ ] 1.2.2 Implement `engine/mesh_builder.py` (greedy meshing)
  - [ ] Face culling: don't emit faces adjacent to solid blocks
  - [ ] Greedy merging to reduce quads
  - [ ] Produce interleaved vertex buffer (pos, uv, light) and index buffer
  - Acceptance: Mesh builder outputs a reduced vertex count vs naive per-face generator for test patterns.

- [ ] 1.2.3 GPU VBO/VAO management in renderer
  - [ ] Upload mesh once per chunk, reuse VBOs when updating
  - [ ] Provide method to delete VBOs when chunk unloads
  - Acceptance: Uploading chunk mesh doesn't reallocate new buffer objects unnecessarily (VBO reuse verified in profiler).

- [ ] 1.2.4 Chunk manager & neighbor queries (`engine/world.py`)
  - [ ] Provide get_chunk(x,z), get_or_generate_chunk(x,z)
  - [ ] Expose world.is_block_solid_global(x,y,z) and neighbor fetching for mesher
  - Acceptance: Mesh builder sees neighbor blocks along chunk boundaries and culls correctly.

- [ ] 1.2.5 Chunk loading/unloading logic
  - [ ] Load chunks in concentric rings around player (breadth-first outward)
  - [ ] Mark chunks outside render distance for unload and free VBOs
  - Acceptance: Player moving causes chunk load/unload without memory leak; debug overlay shows loaded count.

- [ ] 1.2.6 Mark chunk dirty and rebuild flow
  - [ ] On block change, mark chunk.is_dirty and possibly neighbor.is_dirty if at edge
  - [ ] Limit mesh builds per frame via `MAX_CHUNK_UPDATES_PER_FRAME`
  - Acceptance: Rebuilding is spread over frames and no visible stalls while editing blocks.

### 1.3 World generation (terrain)
- [ ] 1.3.1 Implement `world/terrain_generator.py` basic heightmap
  - [ ] Use `noise` Perlin/Simplex with multiple octaves
  - [ ] Generate base layers (bedrock, stone, dirt, grass)
  - Acceptance: Generated terrain looks varied and repeatable with seed.

- [ ] 1.3.2 Chunk-on-demand generation
  - [ ] Generate content when `get_or_generate_chunk` called
  - [ ] Save generated chunks to disk cache (optional)
  - Acceptance: Walking around spawns generated chunks on the fly, consistent with seed.

### 1.4 Physics & collision (foundation)
- [ ] 1.4.1 Implement basic AABB physics skeleton (`mechanics/physics.py`)
  - [ ] Player velocity integration, gravity, jump handling
  - [ ] Per-axis collision resolution (separate X, Y, Z steps)
  - Acceptance: Player doesn't fall through floor and collides with blocks.

- [ ] 1.4.2 Small-test passes for collision at chunk boundaries
  - [ ] Tests for movement across chunk seams
  - Acceptance: Player moves across chunk edges with correct collision.

---

## Phase 2 — Block Interaction & Mining

### 2.1 Raycasting & block selection
- [ ] 2.1.1 Implement DDA raycast (`mechanics/raycast.py`)
  - [ ] Return (block_pos, face_normal) or (None, None)
  - [ ] Respect `REACH_DISTANCE` and ignore non-solid blocks
  - Acceptance: Unit tests cover corner cases and the raycast finds the intended block in examples.

- [ ] 2.1.2 Visual highlight & crosshair feedback
  - [ ] Render selected face highlight or wireframe
  - Acceptance: Highlight matches the block and face returned by raycast.

### 2.2 Block types & registry
- [ ] 2.2.1 Implement `blocks/block.py` base class and registry
  - [ ] Attributes: solid, hardness, tool_requirement, drop, light
  - [ ] Register function to map id↔name↔properties
  - Acceptance: Block registry returns correct properties for sample blocks.

- [ ] 2.2.2 Add core block set
  - [ ] Grass, dirt, stone, bedrock, wood, planks, cobblestone, leaves, glass
  - [ ] Provide texture UV mapping into atlas
  - Acceptance: Blocks render with assigned textures.

### 2.3 Mining & tools
- [ ] 2.3.1 Implement mining progress + tool multipliers (`mechanics/mining.py`)
  - [ ] Breaking takes time; show a progress bar or overlay
  - [ ] Tool effectiveness changes break time; empty hand works but slower
  - Acceptance: Break time matches expected values in tests; progress UI appears.

- [ ] 2.3.2 Implement item drops & item entities
  - [ ] Spawn item entities on break; items can be picked up
  - [ ] Item merging & despawn timer
  - Acceptance: Break a block → item entity appears and can be picked up by player.

### 2.4 Block placement
- [ ] 2.4.1 Placement rules and face handling
  - [ ] Can't place inside player; check collision before placement
  - [ ] Placement attaches to selected face
  - Acceptance: Placing block in front of player works and doesn't trap player.

- [ ] 2.4.2 Update chunk boundaries & dirty flags
  - [ ] When placing on edge, mark neighbor chunk dirty if needed
  - Acceptance: Neighbor chunk meshes update correctly after edge placement.

---

## Phase 3 — UI, Inventory & Hotbar

### 3.1 HUD & hotbar
- [ ] 3.1.1 Crosshair & FPS counter
  - [ ] F3 toggles debug overlay (FPS, position, loaded chunks)
  - Acceptance: F3 shows overlay with working values.

- [ ] 3.1.2 Hotbar implementation (`ui/hotbar.py`)
  - [ ] 9-slot hotbar that maps to number keys 1–9
  - [ ] Visual highlight on selected slot
  - Acceptance: Press number keys to change selection and place blocks accordingly.

### 3.2 Inventory UI
- [ ] 3.2.1 Inventory grid UI (`ui/inventory.py`)
  - [ ] Open/close with E, drag & drop between slots
  - [ ] Stack management and tooltips
  - Acceptance: Can move items freely and stacks respect max stack size.

### 3.3 Item system
- [ ] 3.3.1 Item class & rendering
  - [ ] 2D sprite icons for inventory; simple 3D drop rendering optional
  - Acceptance: Inventory shows icons; dropped items visible in world.

---

## Phase 4 — Crafting & Tools

### 4.1 Crafting system
- [ ] 4.1.1 Implement `mechanics/crafting.py` with `Recipe` dataclass
  - [ ] Shaped & shapeless matching (`match_crafting_grid`)
  - [ ] Recipe registry and serialization
  - Acceptance: Tests for recipe matches (2x2 and 3x3) pass.

- [ ] 4.1.2 Crafting UI (crafting table & player grid)
  - [ ] 2x2 player crafting and 3x3 crafting table
  - [ ] Result preview updates when grid changes
  - Acceptance: Crafting result appears instantly when grid matches a recipe.

### 4.2 Tools & durability
- [ ] 4.2.1 Tool durability system and UI hints
  - [ ] Durability reduces per use; tool breaks when durability reaches 0
  - Acceptance: Tools degrade and are removed when durability is zero.

---

## Phase 5 — Entities & Mobs

### 5.1 Entity system foundation
- [ ] 5.1.1 Base `Entity` class and manager
  - [ ] Position/velocity/bbox/health, update & render hooks
  - Acceptance: Simple entity spawns and updates.

### 5.2 Item Entities (dropped items)
- [ ] 5.2.1 Implement item entity physics and pickup
  - [ ] Magnet pickup to player within small radius
  - Acceptance: Dropped items are collectible and merge.

### 5.3 Friendly mobs & AI
- [ ] 5.3.1 Implement simple mobs (Cow, Chicken, Sheep)
  - [ ] Simple voxel or box model rendering
  - [ ] Basic AI states (wander, idle, flee)
  - Acceptance: Mobs spawn and wander; basic interactions (drops) work.

- [ ] 5.3.2 Mob spawning & despawning
  - [ ] Spawn in valid locations and despawn far away
  - Acceptance: Mob population remains stable and doesn't crash.

---

## Phase 6 — World Features, Lighting, Audio, Particles

### 6.1 Trees, ores, caves
- [ ] 6.1.1 Procedural trees and structures
  - [ ] Place trees after terrain generation; ensure no collision with terrain
  - Acceptance: Trees appear in biomes and persist across reloads.

- [ ] 6.1.2 Ore distribution
  - [ ] Add coal, iron, gold, diamond veins with depth parameters
  - Acceptance: Ores spawn at expected frequencies and depths.

### 6.2 Lighting (incremental)
- [ ] 6.2.1 Implement simple sunlight + block light flood-fill
  - [ ] Use queue to propagate light; process limited updates per frame
  - Acceptance: Basic lighting looks correct and updates on block changes.

- [ ] 6.2.2 Optional: smooth lighting / AO
  - Acceptance: Optional; mark as low priority if performance suffers.

### 6.3 Audio & particle system
- [ ] 6.3.1 Add audio manager and basic sounds (break, place, pickup)
  - Acceptance: Sounds play with volume control.

- [ ] 6.3.2 Implement particle system for block break & pickup
  - Acceptance: Particle effects appear and are performant.

---

## Phase 7 — Save/Load & Optimization

### 7.1 Save system
- [ ] 7.1.1 Chunk-based save format (binary + compression)
  - [ ] Write to temp file and rename atomically
  - Acceptance: Saves are reliable; loading restores world seed and chunk contents.

- [ ] 7.1.2 Player save (position, inventory, world metadata)
  - Acceptance: Player state restores on load.

### 7.2 Performance optimization
- [ ] 7.2.1 Profile hotspots with cProfile
  - [ ] Identify mesh, physics, and rendering bottlenecks
  - Acceptance: Profiling reproduces problem areas and points to concrete fixes.

- [ ] 7.2.2 Optimize mesh generation (Numba/Cython as needed)
  - [ ] Consider moving greedy meshing into Numba-accelerated functions
  - Acceptance: Measured improvement in mesh generation time for worst-case chunks.

- [ ] 7.2.3 Rendering optimizations
  - [ ] Frustum culling per-chunk
  - [ ] VBO reuse and pooling
  - [ ] Optional LOD for distant chunks
  - Acceptance: Rendering load reduces and FPS improves at target render distances.

### 7.3 Testing & fixes
- [ ] 7.3.1 Regression tests for previously fixed bugs
  - [ ] Add test for chunk edge rebuilds and lighting updates
  - Acceptance: Tests prevent regressions.

---

## Phase 8 — Polish & Release

### 8.1 Menus, settings, polish
- [ ] 8.1.1 Pause menu, settings (render distance, controls)
- [ ] 8.1.2 World selection & create world UI
- [ ] 8.1.3 Creative mode toggle for testing
  - Acceptance: Menus functional and settings persist.

### 8.2 Docs & README
- [ ] 8.2.1 Complete README with setup & run instructions
- [ ] 8.2.2 Developer notes for architecture & algorithm choices
  - Acceptance: New developer can get project running with README steps.

### 8.3 Final testing
- [ ] 8.3.1 Playtest pass and fix critical bugs
- [ ] 8.3.2 Performance acceptance testing (target: 60 FPS at render distance 8)
  - Acceptance: Core features stable and performance targets met or documented deviations.

---

## Priority & Emergency Cut List (copy from quick reference)
If time runs out, cut features in this order:
1. Breeding system
2. Audio/sound effects
3. Lighting system (use ambient only)
4. Furnace/smelting
5. Particle effects
6. Multiple biomes
7. Trees
8. Mobs

---

## Quick references & tests to run regularly
- Run `pytest` before pushing significant changes.
- Use the debug overlay (F3) to check loaded chunks, FPS, memory.
- Keep `MAX_CHUNK_UPDATES_PER_FRAME` conservative while developing.

---

## Notes & Next Steps
- Add links to related issues or commit hashes when completing tasks.
- Consider adding a `project_board.md` or GitHub project board that mirrors this checklist for visual tracking.

---

*Generated from `plan.md` and `QUICK_REFERENCE.md`.*

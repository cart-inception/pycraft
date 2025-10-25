# Quick Reference Guide - Pycraft Development

## Critical "Don't Forget" List

### 🔴 **MUST DO - Or Project Will Fail**

1. **Use Modern OpenGL**
   - ❌ NO: `glBegin()`, `glEnd()`, `glVertex3f()`
   - ✅ YES: VAO, VBO, shaders
   
2. **Implement Face Culling**
   - Don't render faces between solid blocks
   - Reduces vertices by 80%+
   
3. **Generate Mesh Per Chunk, Not Per Block**
   - One draw call per chunk, not per block
   - Critical for performance
   
4. **Cache Generated Meshes**
   - Only rebuild when chunk changes
   - Mark chunks as "dirty" when modified

5. **Profile Early and Often**
   - Don't wait until the end
   - Use cProfile to find bottlenecks

### ⚠️ **HIGH PRIORITY**

6. **Implement Proper Raycasting**
   - Use DDA algorithm (in plan)
   - Don't use brute force iteration

7. **Handle Chunk Boundaries Correctly**
   - Query neighbor chunks for face culling
   - Rebuild neighbors when edge blocks change

8. **Separate Physics Axes**
   - Handle X, Y, Z collisions separately
   - Prevents stuck-in-wall bugs

9. **Use NumPy for All Array Operations**
   - Block storage: `np.zeros((16,256,16), dtype=np.uint8)`
   - Much faster than Python lists

10. **Implement Save File Safety**
    - Write to temp file first
    - Rename atomically
    - Prevents corruption

### 💡 **GOOD PRACTICES**

11. **Centralize Configuration**
    - All magic numbers in `config.py`
    - Makes tuning much easier

12. **Implement Debug Mode**
    - Show FPS, position, chunk info
    - Toggle with F3 key

13. **Add Creative Mode Early**
    - Flying, instant break, infinite items
    - Makes testing much faster

14. **Use Logging, Not Print**
    - Proper logging levels
    - Can disable debug logs in production

15. **Test on Minimal Hardware**
    - If it works on low-end, works everywhere
    - Don't just test on your gaming PC

## Common Mistakes & Quick Fixes

### Mistake: Black Screen
**Fix:**
- Check shader compilation errors
- Enable OpenGL debug output
- Verify VAO/VBO setup
- Check glGetError() after each call

### Mistake: Terrible FPS
**Fix:**
- Verify using VBOs, not immediate mode
- Implement face culling
- Check if regenerating meshes every frame
- Enable frustum culling

### Mistake: Player Falls Through Floor
**Fix:**
- Use continuous collision detection
- Check collision multiple times per frame if moving fast
- Separate Y-axis collision from XZ

### Mistake: Chunks Don't Connect
**Fix:**
- Query neighbor chunks when building mesh
- Rebuild neighbor chunks when edge blocks change
- Handle null neighbors at world edge

### Mistake: Memory Usage Growing
**Fix:**
- Unload distant chunks
- Delete VBOs when unloading chunks
- Use uint8 for block IDs, not int

### Mistake: Save Files Huge
**Fix:**
- Only save modified chunks
- Use binary format, not JSON
- Enable compression (gzip)

## Development Workflow

### Phase 0: Setup (Week 1)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pygame PyOpenGL PyOpenGL-accelerate numpy pillow noise

# Create requirements.txt
pip freeze > requirements.txt

# Setup git
git init
git add .
git commit -m "Initial commit"
```

### Phase 1: First Rendering (Week 2)
**Goal:** Single colored cube on screen with moving camera

**Test:** Can you move around the cube with WASD and mouse?

### Phase 2: Chunk System (Week 3)
**Goal:** Simple terrain with multiple chunks

**Test:** Can you walk around terrain that loads as you move?

### Phase 3: Block Interaction (Weeks 4-5)
**Goal:** Break and place blocks

**Test:** Can you break blocks and place them elsewhere?

### Weekly Check-In Questions
1. Does it run at 60 FPS?
2. Are there any crashes?
3. Does it feel responsive?
4. Can you test the new features?
5. What's the biggest current problem?

## Performance Targets

| Metric | Target | Critical |
|--------|--------|----------|
| FPS | 60 | >30 |
| Mesh Generation | <16ms | <100ms |
| Memory Usage | <2GB | <4GB |
| Chunk Load Time | <100ms | <500ms |
| Save/Load Time | <5s | <30s |

## Testing Checklist

### Before Each Commit:
- [ ] No crashes in normal gameplay
- [ ] FPS above 30
- [ ] No Python exceptions in console
- [ ] Changes tested manually

### Before Each Milestone:
- [ ] All features from phase work
- [ ] No known critical bugs
- [ ] Performance targets met
- [ ] Code committed to git

## File Priorities

### Implement First:
1. `engine/renderer.py` - Basic rendering
2. `engine/camera.py` - Camera movement
3. `engine/chunk.py` - Chunk data structure
4. `engine/mesh_builder.py` - Mesh generation
5. `world/terrain_generator.py` - Terrain generation

### Implement Second:
6. `mechanics/physics.py` - Player physics
7. `mechanics/raycast.py` - Block selection
8. `mechanics/mining.py` - Block breaking
9. `ui/inventory.py` - Inventory system
10. `ui/hotbar.py` - Quick access bar

### Implement Last:
11. `entities/mobs.py` - Mob system
12. `entities/ai.py` - Mob AI
13. `mechanics/crafting.py` - Crafting system
14. Audio, particles, lighting

## Emergency Cuts (If Running Out of Time)

Cut in this order:
1. Breeding system
2. Audio/sound effects
3. Lighting system (use simple ambient only)
4. Furnace/smelting
5. Particle effects
6. Multiple biomes (just use plains)
7. Trees (just flat terrain)
8. Mobs (focus on blocks only)

**Minimum Viable Product:**
- Voxel terrain you can walk in
- Break and place blocks
- Basic inventory
- Simple crafting (wood → planks → sticks → tools)
- Saves/loads

## Useful Debug Commands

Implement these early:
```python
# Debug commands (console or key bindings)
/teleport x y z          # Teleport player
/give item count         # Add items to inventory
/fly                     # Toggle fly mode
/time set 1200           # Set time of day
/fill x1 y1 z1 x2 y2 z2  # Fill area with blocks
/clear                   # Clear inventory
/speed 2.0               # Change movement speed
```

## Resource Links

- **OpenGL Tutorial:** https://learnopengl.com (use sections on VAO/VBO/Shaders)
- **PyOpenGL Documentation:** http://pyopengl.sourceforge.net
- **Perlin Noise Explanation:** https://adrianb.io/2014/08/09/perlinnoise.html
- **Greedy Meshing:** Search "voxel greedy meshing algorithm"
- **DDA Raycasting:** "Fast Voxel Traversal Algorithm" by Amanatides & Woo

## Final Reminders

✅ **Start simple** - Get basics working before adding features  
✅ **Profile early** - Don't guess where slowness is  
✅ **Test often** - Small bugs become big problems  
✅ **Commit frequently** - Easy to roll back if needed  
✅ **Ask for help** - If stuck >2 hours, seek assistance  

❌ **Don't over-engineer** - YAGNI (You Ain't Gonna Need It)  
❌ **Don't skip testing** - Manual testing at minimum  
❌ **Don't add features early** - Core features first  
❌ **Don't ignore performance** - Profile regularly  
❌ **Don't use immediate mode OpenGL** - This cannot be stressed enough!  

---

**Remember:** A simple game that runs well is better than a complex game that doesn't run at all!

Good luck! 🚀

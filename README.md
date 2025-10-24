# Pycraft - Minecraft Clone in Python

A voxel-based sandbox game built with Python, Pygame, and modern OpenGL. This project implements a Minecraft-like game with terrain generation, block interaction, and crafting mechanics.

## Features

### Currently Implemented (Phase 0)
- ✅ Modern OpenGL rendering pipeline (VAO/VBO system)
- ✅ Centralized configuration system
- ✅ Shader management with GLSL support
- ✅ Testing framework with pytest
- ✅ Comprehensive project structure
- ✅ Performance monitoring utilities

### Planned Features (Future Phases)
- 🏗️ Terrain generation with Perlin noise
- 🏗️ Chunk-based world loading/unloading
- 🏗️ Player physics and collision detection
- 🏗️ Block breaking and placement
- 🏗️ Inventory and crafting system
- 🏗️ Mobs and entities
- 🏗️ Lighting system
- 🏗️ Save/load functionality

## Requirements

- Python 3.8 or higher
- OpenGL 3.3 compatible graphics card
- 4GB+ RAM recommended
- Development platform: Linux, Windows, or macOS

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd pycraft
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Run tests to verify everything is working
pytest

# Run the game (should show a blank window)
python main.py
```

## Project Structure

```
pycraft/
├── main.py                 # Game entry point
├── config.py               # Central configuration
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── QUICK_REFERENCE.md     # Development quick reference
├── plan.md               # Detailed development plan
├── tasks.md              # Task tracking checklist
├── .gitignore            # Git ignore patterns
├── conftest.py           # Pytest configuration
├── engine/               # Core rendering engine
│   ├── __init__.py
│   ├── shader.py         # Shader management system
│   └── utils.py          # Math and utility functions
├── shaders/              # GLSL shader files
│   ├── vertex.glsl       # Vertex shader
│   └── fragment.glsl     # Fragment shader
├── blocks/               # Block system
│   └── __init__.py
├── entities/             # Player and mobs
│   └── __init__.py
├── ui/                   # User interface
│   └── __init__.py
├── mechanics/            # Game mechanics
│   └── __init__.py
├── world/                # Terrain generation
│   └── __init__.py
├── audio/                # Sound system
│   └── __init__.py
├── tests/                # Test suite
│   ├── __init__.py
│   ├── test_config.py
│   └── test_utils.py
├── saves/                # Save game directory (created automatically)
└── resources/            # Generated resources
    └── textures/         # Procedurally generated textures
```

## Controls

### Current Controls (Phase 0)
- `ESC` - Exit game
- `F3` - Toggle debug mode
- `F11` - Toggle fullscreen

### Planned Controls
- `W/A/S/D` - Movement
- `Mouse` - Camera look
- `Space` - Jump
- `Shift` - Sneak
- `Ctrl` - Sprint
- `E` - Inventory
- `1-9` - Hotbar selection
- `Left Click` - Break block
- `Right Click` - Place block

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_config.py

# Run tests with verbose output
pytest -v
```

### Code Quality
This project follows PEP 8 style guidelines and uses type hints where appropriate.

### Configuration
All game settings are centralized in `config.py`. Key settings include:

- `WINDOW_WIDTH`, `WINDOW_HEIGHT` - Display resolution
- `CHUNK_SIZE`, `CHUNK_HEIGHT` - World chunk dimensions
- `RENDER_DISTANCE` - How many chunks to render
- `PLAYER_SPEED`, `GRAVITY` - Physics parameters
- `DEBUG_MODE` - Enable debug overlay and features

### Adding New Features
1. Update the appropriate module in the project structure
2. Add configuration options to `config.py` if needed
3. Write tests for new functionality
4. Update documentation

## Performance Considerations

This project is designed with performance in mind:

- **Modern OpenGL**: Uses VAO/VBO rendering instead of immediate mode
- **Greedy Meshing**: Reduces vertex count by merging adjacent block faces
- **Face Culling**: Doesn't render faces between solid blocks
- **Chunk-based Loading**: Only loads chunks near the player
- **NumPy Optimizations**: Uses efficient array operations
- **Threading**: Supports parallel mesh generation

## Debug Features

When `DEBUG_MODE` is enabled in `config.py`:

- FPS counter in window title
- OpenGL error checking
- Performance timing for critical operations
- Memory usage monitoring
- Debug overlay with player position and world info

## Architecture Decisions

### Modern OpenGL (Not Immediate Mode)
This project uses modern OpenGL (3.3+) with VAOs, VBOs, and shaders. This provides:
- Better performance
- Hardware acceleration
- Flexibility for advanced effects
- Future-proofing

### Chunk-based World System
- 16x256x16 block chunks (configurable)
- Greedy meshing for performance
- On-demand generation
- Efficient memory usage

### Modular Design
Clear separation of concerns:
- `engine/` - Core rendering and utilities
- `world/` - Terrain generation and world management
- `blocks/` - Block types and properties
- `entities/` - Player and mob system
- `mechanics/` - Game logic and physics
- `ui/` - User interface components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation
7. Submit a pull request

## Known Issues

- None currently (Phase 0 foundation complete)

## Roadmap

### Phase 0 ✅ (Complete)
- Project setup and foundation
- Basic rendering pipeline
- Configuration system
- Testing framework

### Phase 1 (In Progress)
- Basic 3D rendering and camera
- Chunk data and mesh building
- World generation
- Basic physics

### Phase 2 (Planned)
- Block interaction and mining
- Raycasting for block selection
- Block types and properties
- Block placement

### Phase 3 (Planned)
- UI and inventory system
- Hotbar implementation
- Item system

### Phase 4+ (Planned)
- Crafting system
- Entities and mobs
- World features and lighting
- Save/load and optimization

## Troubleshooting

### Common Issues

**Black screen on startup**
- Check OpenGL version: `python -c "import OpenGL; print(OpenGL.GL.glGetString(OpenGL.GL.GL_VERSION))"`
- Update graphics drivers
- Ensure OpenGL 3.3+ support

**Import errors**
- Ensure virtual environment is activated
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

**Performance issues**
- Reduce `RENDER_DISTANCE` in config.py
- Ensure `DEBUG_MODE` is False for production
- Check if vsync is causing issues

**Shader compilation errors**
- Check shader files in `shaders/` directory
- Verify OpenGL version compatibility
- Check shader logs in console output

### Getting Help

1. Check the [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for development tips
2. Review the [plan.md](plan.md) for technical details
3. Check the [tasks.md](tasks.md) for current development progress
4. Enable debug mode (`F3`) to see performance information
5. Check the log files (`game.log`) for error details

## License

This project is open source. See LICENSE file for details.

## Credits

Based on modern OpenGL best practices and voxel game design patterns. Inspired by Minecraft and various open-source voxel engines.
# 3D Character Models Directory

This directory is a placeholder for future 3D character models that will be used in the Gemini Voice Interface.

## Future Implementation

When implementing 3D characters, you can place model files in this directory. The web interface is already prepared with:

1. A canvas element for rendering 3D models
2. Basic character animation hooks for speaking and idle states
3. WebGL support detection

## Recommended File Types

For web-based 3D characters, the following file types are recommended:

- `.glb` or `.gltf` - GL Transmission Format (efficient for web)
- `.obj` - Wavefront OBJ (with accompanying `.mtl` files)
- `.fbx` - Filmbox (may require conversion for web use)

## Implementation Tips

When implementing 3D characters:

1. Use a library like Three.js or Babylon.js for rendering
2. Keep models optimized for web performance (low poly count, efficient textures)
3. Implement at minimum these animations:
   - Idle state
   - Speaking/talking
   - Reactions to different types of messages

## Example Structure

A typical implementation might include:

```
models/
  ├── pneuma/                  # Character name
  │   ├── model.glb           # Main model file
  │   ├── textures/           # Texture files
  │   │   ├── diffuse.png
  │   │   ├── normal.png
  │   │   └── ...
  │   └── animations/         # Animation files
  │       ├── idle.glb
  │       ├── talking.glb
  │       └── ...
  └── README.md               # This file
```

## Integration

The `character.js` file in the parent directory contains placeholder code for character initialization and animation. When implementing 3D characters, update this file with actual rendering and animation code.

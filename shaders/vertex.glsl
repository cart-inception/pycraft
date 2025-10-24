#version 330 core

// Vertex inputs
layout(location = 0) in vec3 aPosition;     // World position
layout(location = 1) in vec2 aTexCoord;     // Texture coordinates
layout(location = 2) in float aLighting;    // Lighting level (0.0-1.0)

// Uniforms
uniform mat4 uProjection;   // Projection matrix
uniform mat4 uView;         // View matrix
uniform mat4 uModel;        // Model matrix

// Outputs to fragment shader
out vec2 vTexCoord;
out float vLighting;

void main()
{
    // Transform vertex position
    gl_Position = uProjection * uView * uModel * vec4(aPosition, 1.0);

    // Pass through texture coordinates and lighting
    vTexCoord = aTexCoord;
    vLighting = aLighting;
}
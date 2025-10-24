#version 330 core

// Inputs from vertex shader
in vec2 vTexCoord;
in float vLighting;

// Output color
out vec4 FragColor;

// Uniforms
uniform sampler2D uTexture;   // Texture atlas
uniform vec3 uAmbientColor;  // Ambient light color
uniform vec3 uSunColor;      // Sunlight color
uniform float uTime;         // Game time (for animations)

void main()
{
    // Sample texture
    vec4 texColor = texture(uTexture, vTexCoord);

    // Discard fully transparent pixels (for things like leaves)
    if (texColor.a < 0.1)
        discard;

    // Apply lighting
    vec3 ambient = uAmbientColor * 0.3;           // Base ambient light
    vec3 sunlight = uSunColor * vLighting * 0.7;  // Sunlight contribution
    vec3 finalLight = ambient + sunlight;

    // Apply lighting to texture color
    FragColor = vec4(texColor.rgb * finalLight, texColor.a);
}
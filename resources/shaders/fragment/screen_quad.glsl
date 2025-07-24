// resources/shaders/fragment/screen_quad.glsl
#version 460 core
out vec4 FragColor;
in vec2 TexCoords;
uniform sampler2D screenTexture;

void main() {
    FragColor = texture(screenTexture, TexCoords);
}

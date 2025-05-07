// resources/shaders/fragment/core/model.glsl
#version 330 core
out vec4 FragColor;

// 输入
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
uniform sampler2D texture_diffuse;
uniform sampler2D texture_specular;

void main() {
    vec4 diffuse = texture(texture_diffuse, TexCoords);

    FragColor = diffuse;
}
// resources/shaders/fragment/model/generic.glsl
#version 460 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;
in vec4 FragPosLightSpace;

uniform sampler2D texture_diffuse1;
uniform sampler2D texture_specular;
uniform sampler2D texture_normal;
uniform sampler2D texture_ao;

uniform vec3 viewPos;
uniform mat4 model;

void main() {
    // 丢弃透明片段
    if(texture(texture_diffuse1, TexCoords).a < 0.1) discard;
    
    FragColor = vec4(texture(texture_diffuse1, TexCoords).rgb, 1.0);
}
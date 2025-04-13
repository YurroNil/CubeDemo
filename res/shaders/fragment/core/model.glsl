#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

uniform sampler2D texture_diffuse1;  // 漫反射贴图
uniform sampler2D texture_normal1;   // 法线贴图
uniform sampler2D texture_specular1; // 高光贴图
uniform vec3 defaultColor = vec3(0.8, 0.8, 0.8); // 默认灰色

void main() {
    vec4 diffuse = texture(texture_diffuse1, TexCoords);
    
    // 如果没有漫反射贴图，使用默认颜色
    if (diffuse.a < 0.01) {
        diffuse = vec4(defaultColor, 1.0);
    }
    
    FragColor = diffuse;
}
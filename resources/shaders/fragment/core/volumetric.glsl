// resources/shaders/fragment/core/volumetric.glsl
#version 330 core
out vec4 FragColor;

in vec3 WorldPos;

uniform vec3 lightColor;
uniform vec3 viewPos;

void main() {
    // 基于距离的衰减
    float dist = length(viewPos - WorldPos);
    float attenuation = 1.0 / (1.0 + 0.1*dist + 0.01*dist*dist);
    
    // 边缘衰减（圆锥形衰减）
    vec3 lightDir = normalize(WorldPos - viewPos);
    float edgeFactor = smoothstep(0.8, 1.0, dot(lightDir, vec3(0,1,0)));
    
    // 最终颜色
    vec3 finalColor = lightColor * attenuation * (1.0 - edgeFactor);
    FragColor = vec4(finalColor, 0.3); // 半透明效果
}

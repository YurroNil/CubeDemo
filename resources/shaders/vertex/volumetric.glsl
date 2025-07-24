// resources/shaders/vertex/volumetric.glsl
// resources/shaders/vertex/volumetric.glsl
#version 460 core
layout (location = 0) in vec3 aTip;
layout (location = 1) in vec3 aBase;
layout (location = 2) in float aRadius;

out VS_OUT {
    vec3 tip;
    vec3 base;
    float radius;
} vs_out;

uniform mat4 model; // 添加模型矩阵

void main() {
    // 应用模型矩阵变换
    vs_out.tip = vec3(model * vec4(aTip, 1.0));
    vs_out.base = vec3(model * vec4(aBase, 1.0));
    vs_out.radius = aRadius;
    
    // 使用tip位置作为gl_Position
    gl_Position = vec4(vs_out.tip, 1.0);
}

// resources/shaders/vertex/core/volumetric.glsl
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;
layout (location = 3) in vec3 aEmitColor;

out vec3 WorldPos;
out vec3 Normal;
out vec3 EmitColor;
out float DistToAxis;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    WorldPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    EmitColor = aEmitColor;
    
    // 计算到光束中心轴的距离 (用于径向衰减)
    DistToAxis = length(aPos.xz);
    
    gl_Position = projection * view * vec4(WorldPos, 1.0);
}
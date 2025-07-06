// resources/shaders/geometry/core/volumetric.glsl
#version 450 core
layout (points) in;
layout (triangle_strip, max_vertices = 128*3) out;

in VS_OUT {
    vec3 tip;
    vec3 base;
    float radius;
} gs_in[];

out vec3 WorldPos;
out vec3 Normal;
out vec3 EmitColor;
out float DistToAxis;

uniform mat4 view;
uniform mat4 projection;
uniform int segments = 64;

void main() {
    vec3 tip = gs_in[0].tip;
    vec3 base = gs_in[0].base;
    float radius = gs_in[0].radius;
    
    // 计算圆锥轴
    vec3 coneAxis = tip - base;
    float coneLength = length(coneAxis);
    vec3 coneDir = coneAxis / coneLength;
    
    // 创建正交基
    vec3 up = abs(coneDir.y) > 0.999 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(coneDir, up));
    vec3 forward = normalize(cross(right, coneDir));
    
    // 生成圆锥侧面
    for (int i = 0; i < segments; ++i) {
        float angle1 = 2.0 * 3.1415926 * float(i) / float(segments);
        float angle2 = 2.0 * 3.1415926 * float(i+1) / float(segments);
        
        // 计算圆周点
        vec3 offset1 = radius * (cos(angle1) * right + sin(angle1) * forward);
        vec3 offset2 = radius * (cos(angle2) * right + sin(angle2) * forward);
        
        vec3 worldPos1 = base + offset1;
        vec3 worldPos2 = base + offset2;
        
        // 计算法线
        vec3 normal1 = normalize(worldPos1 - base);
        vec3 normal2 = normalize(worldPos2 - base);
        vec3 tipNormal = coneDir;
        
        // 发射顶点1 (圆锥顶点)
        WorldPos = tip;
        Normal = tipNormal;
        EmitColor = vec3(1.0);
        DistToAxis = 0.0;
        gl_Position = projection * view * vec4(tip, 1.0);
        EmitVertex();
        
        // 发射顶点2 (底面点1)
        WorldPos = worldPos1;
        Normal = normal1;
        EmitColor = vec3(1.0);
        DistToAxis = length(offset1);
        gl_Position = projection * view * vec4(worldPos1, 1.0);
        EmitVertex();
        
        // 发射顶点3 (底面点2)
        WorldPos = worldPos2;
        Normal = normal2;
        EmitColor = vec3(1.0);
        DistToAxis = length(offset2);
        gl_Position = projection * view * vec4(worldPos2, 1.0);
        EmitVertex();
        
        EndPrimitive();
    }
}

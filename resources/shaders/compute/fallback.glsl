// resources/shaders/compute/fallback.glsl
#version 460 core

// 该着色器用于测试CPU端的代码问题，输出纯红色图像

layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba32f, binding = 0) uniform image2D outputTex;
uniform vec3 cameraPosition;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    vec3 color = vec3(1.0, 0.0, 0.0); // 设置为纯红色
    imageStore(outputTex, pixel, vec4(color, 1.0));
}

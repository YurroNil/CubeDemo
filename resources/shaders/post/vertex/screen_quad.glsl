// resources/shaders/post/vertex/screen_quad.glsl
#version 460 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main() {
    // 直接传递位置为NDC坐标（范围[-1,1]）
    gl_Position = vec4(aPos, 0.0, 1.0);
    
    // 传递纹理坐标给片段着色器
    TexCoords = aTexCoords;
}

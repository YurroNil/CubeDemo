// include/rendering/renderer.h

#pragma once
#include "rendering/shader.h"
#include "rendering/mesh.h"
#include "core/camera.h"

struct GLFWwindow;

class Renderer {
public:
    static void Init();  // 初始化渲染器（设置OpenGL状态）
    static void BeginFrame();  // 开始一帧的渲染（清空缓冲区）
    
    // 提交一个可渲染对象（Mesh+Shader）
    static void Submit(
        const Shader& shader,
        const Mesh& mesh,
        const glm::mat4& model = glm::mat4(1.0f)
    );
    
    // 应用相机参数到着色器
    static void ApplyCamera(
        const Shader& shader,
        const Camera camera,
        float aspectRatio
    );
    
    // 结束一帧的渲染（交换缓冲区）
    static void EndFrame(GLFWwindow* window);
};
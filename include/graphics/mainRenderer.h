// include/graphics/mainRenderer.h

#pragma once
#include "graphics/shader.h"
#include "graphics/mesh.h"
#include "core/camera.h"

struct GLFWwindow;

class Renderer {
public:
    static void Init();  // 初始化渲染器（设置OpenGL状态）
    static void BeginFrame();  // 开始一帧的渲染（清空缓冲区）

    
    // 结束一帧的渲染（交换缓冲区）
    static void EndFrame(GLFWwindow* window);
};
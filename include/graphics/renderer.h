// include/graphics/renderer.h

#pragma once
#include "kits/glfw.h"

namespace CubeDemo {

class Renderer {
public:
    static void Init();  // 初始化渲染器（设置OpenGL状态）
    static void BeginFrame();  // 开始一帧的渲染（清空缓冲区）
    
    // 结束一帧的渲染（交换缓冲区）
    static void EndFrame(GLFWwindow* window);
};

}
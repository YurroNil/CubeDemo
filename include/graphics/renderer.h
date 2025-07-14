// include/graphics/renderer.h
#pragma once

namespace CubeDemo {

// 向前声明
class RayTracing; class Shader;

class Renderer {
// private
    inline static Shader* m_QuadShader = nullptr;

public:
    inline static RayTracing* s_RayTracing = nullptr;
    inline static bool s_isFirstLoad = true;

    static void Init();  // 初始化渲染器（设置OpenGL状态）
    static void Cleanup();
    static void BeginFrame();  // 开始一帧的渲染（清空缓冲区）
    static void RayTracingEnabled(bool enabled = true);
    // 结束一帧的渲染（交换缓冲区）
    static void EndFrame(GLFWwindow* window);
    static void RenderFullscreenQuad();
    static void Create_RT_Inst(bool is_scene_switching = false);

};
}   // namespace CubeDemo

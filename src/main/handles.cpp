// src/main/handles.cpp
#include "pch.h"
#include "main/handles.h"

namespace CubeDemo {

// 输入管理
void handle_input(GLFWwindow* window, Camera* camera) {
    INPUTS::ProcPanelKeys(window);
    INPUTS::ProcCameraKeys(window, camera, TIME::GetDeltaTime());
}
// 开始帧
void begin_frame(Camera* camera) {
    Renderer::BeginFrame();
    TIME::Update();
}

// 结束帧
void end_frame_handling(GLFWwindow* window) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}

// 输入窗口设置
void handle_window_settings(GLFWwindow* window) {
    WINDOW::UpdateWinSize(window);       // 更新窗口尺寸
    WINDOW::FullscreenTrigger(window);   // 全屏
}
}   // namespace CubeDemo

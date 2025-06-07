// src/main/handles.cpp
#include "pch.h"
#include "main/handles.h"
#include "core/camera.h"

namespace CubeDemo {

// 输入管理
void handle_input(GLFWwindow* window) {
    Inputs::isEscPressed(window);
    if (!Inputs::isGamePaused) Inputs::ProcKeyboard(window, Time::DeltaTime());
}

// 开始帧
void begin_frame(Camera* camera) {
    Renderer::BeginFrame();
    Time::Update();
    UIMng::RenderLoop(Window::GetWindow(), camera);
}

// 结束帧
void end_frame_handling(GLFWwindow* window) {
    Renderer::EndFrame(window);
    glfwPollEvents();
}

// 输入窗口设置
void handle_window_settings(GLFWwindow* window) {
    Window::UpdateWinSize(window);       // 更新窗口尺寸
    Window::FullscreenTrigger(window);   // 全屏
}

}   // namespace CubeDemo

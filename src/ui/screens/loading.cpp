// src/ui/screens/loading.cpp
#include "pch.h"
#include "ui/screens/loading.h"
#include "imgui.h"
#include "loaders/model_initer.h"

namespace CubeDemo::UI {

void LoadingScreen::Init() {
    if (s_Inited) return;

    // 初始化显示尺寸
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(
        static_cast<float>(WINDOW::GetWidth()), 
        static_cast<float>(WINDOW::GetHeight())
    );
    // 禁用INI文件保存
    io.IniFilename = nullptr;
    s_Inited = true;
}

void LoadingScreen::Cleanup() {
    s_Inited = false; s_isLoading = false;
}
// 渲染静态的加载画面(同步模式专用)
void LoadingScreen::StaticGraphic() {
    // 初始化阶段需额外beginframe
    if(MIL::s_isInitPhase) Renderer::BeginFrame();

    // 处理事件
    glfwPollEvents();
    
    // 清除屏幕 - 使用深色背景
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // 确保使用默认帧缓冲区
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, WINDOW::GetWidth(), WINDOW::GetHeight());
    
    // 更新显示尺寸
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(
        static_cast<float>(WINDOW::GetWidth()), 
        static_cast<float>(WINDOW::GetHeight())
    );
    
    // 渲染文字
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(io.DisplaySize);
    
    if (ImGui::Begin("Loading Screen", nullptr, 
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoInputs))
    {
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();

        ImGui::SetCursorPos(ImVec2(center.x, center.y + 20));
        ImGui::Text("Loading...");
        ImGui::End();
    }

    // 初始化阶段则endframe
    if(MIL::s_isInitPhase) Renderer::EndFrame(WINDOW::GetWindow());
}

// 渲染静态的加载画面(异步模式专用)
void LoadingScreen::DynamicGraphic() {}

void LoadingScreen::Render(bool async_mode) {
    if (!s_Inited) return;
    if(!s_isLoading) s_isLoading = true;

    if(async_mode) DynamicGraphic();
    else StaticGraphic();
}
} // namespace CubeDemo::UI

// src/ui/uiMng.cpp
#include "pch.h"
// UI模块
#include "ui/uiMng.h"
#include "ui/panels/pause.h"
#include "ui/panels/control.h"
#include "ui/panels/debug.h"
// 核心模块
#include "core/inputs.h"
#include "core/window.h"
#include "core/camera.h"
// 加载器模块
#include "loaders/font.h"

namespace CubeDemo {

// 初始化UI管理器
void UIMng::Init() {
    // 初始化ImGui库
    InitImGui();
    // 配置ImGui的样式
    ConfigureImGuiStyle();
    // 加载自定义字体
    FL::LoadFonts();
}

// 渲染循环，用于在每一帧中更新和渲染UI
void UIMng::RenderLoop(GLFWwindow* window, Camera* camera) {
    UI::PausePanel::Render(window);    // 处理暂停菜单逻辑

    if (Inputs::s_isDebugVisible) {
        UI::ControlPanel::Render(camera); // 控制面板
        UI::DebugPanel::Render(camera);   // 调试面板
    }
}

// 初始化ImGui库
void UIMng::InitImGui() {
    IMGUI_CHECKVERSION(); // 检查ImGui版本
    ImGui::CreateContext(); // 创建ImGui上下文
    ImGui_ImplGlfw_InitForOpenGL(Window::GetWindow(), true); // 初始化ImGui的GLFW后端
    ImGui_ImplOpenGL3_Init("#version 330"); // 初始化ImGui的OpenGL3后端，指定着色器版本
}

// 配置ImGui的样式
void UIMng::ConfigureImGuiStyle() {
    ImGuiStyle& style = ImGui::GetStyle();   // 获取ImGui样式对象
    style.WindowPadding = ImVec2(15, 15);    // 设置窗口内边距
    style.FramePadding = ImVec2(10, 10);     // 设置控件内边距
    style.ItemSpacing = ImVec2(10, 15);      // 设置控件之间的间距
    style.ScaleAllSizes(1.5f);               // 放大所有尺寸以适配高DPI屏幕
    ImGui::StyleColorsDark();                // 使用深色主题
}

// 获取窗口中心位置
ImVec2 UIMng::GetWindowCenter(GLFWwindow* window) {
    Window::UpdateWinSize(window); // 更新窗口尺寸

    return ImVec2(Window::GetWidth()/2.0f, Window::GetHeight()/2.0f); // 返回窗口中心位置
}
}

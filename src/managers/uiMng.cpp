// src/managers/uiMng.cpp
#include "pch.h"
// UI模块
#include "managers/uiMng.h"
#include "ui/panels/pause.h"
#include "ui/panels/control.h"
#include "ui/panels/debug.h"
#include "ui/panels/edit.h"
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
    // 新增编辑面板渲染
    if (Inputs::s_isEditMode) {
        UI::EditPanel::Render();
    }
}

// 初始化ImGui库
void UIMng::InitImGui() {
    IMGUI_CHECKVERSION(); // 检查ImGui版本
    ImGui::CreateContext(); // 创建ImGui上下文
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    ImGui_ImplGlfw_InitForOpenGL(Window::GetWindow(), true); // 初始化ImGui的GLFW后端
    ImGui_ImplOpenGL3_Init("#version 330"); // 初始化ImGui的OpenGL3后端，指定着色器版本
    
}

// 配置ImGui的样式
void UIMng::ConfigureImGuiStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    
    // 圆角
    style.FrameRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.WindowRounding = 8.0f;
    style.PopupRounding = 6.0f;
    
    // 颜色
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
    colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.10f, 0.10f, 0.94f);
    colors[ImGuiCol_Border] = ImVec4(0.25f, 0.25f, 0.25f, 0.50f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.18f, 0.18f, 0.18f, 0.54f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.22f, 0.22f, 0.22f, 0.54f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.26f, 0.26f, 0.26f, 0.54f);
    colors[ImGuiCol_Button] = ImVec4(0.26f, 0.59f, 0.98f, 0.40f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.26f, 0.59f, 0.98f, 0.60f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.06f, 0.53f, 0.98f, 1.00f);
    
    // 间距
    style.WindowPadding = ImVec2(12, 12);
    style.FramePadding = ImVec2(10, 6);
    style.ItemSpacing = ImVec2(10, 8);
    style.ItemInnerSpacing = ImVec2(8, 6);
}

// 获取窗口中心位置
ImVec2 UIMng::GetWindowCenter(GLFWwindow* window) {
    Window::UpdateWinSize(window); // 更新窗口尺寸

    return ImVec2(Window::GetWidth()/2.0f, Window::GetHeight()/2.0f); // 返回窗口中心位置
}
}

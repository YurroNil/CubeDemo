// src/managers/uiMng.cpp
#include "pch.h"
// 管理器模块
#include "managers/uiMng.h"
// UI模块
#include "ui/panels/edit.h"
#include "ui/panels/pause.h"
#include "ui/panels/control.h"
#include "ui/panels/debug.h"
#include "ui/panels/presetlib.h"
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

// 渲染循环的初始化. 在完成了程序初始化阶段后，进入渲染循环的第一帧，完成第一次初始化, 仅执行一次.
void UIMng::RenderInit() {
    // 编辑面板初始化
    UI::EditPanel::Init();
     // 预设库面板初始化
    UI::PresetlibPanel::Init();
}

// 渲染循环，用于在每一帧中更新和渲染UI
void UIMng::RenderLoop(GLFWwindow* window, Camera* camera) {
    if (INPUTS::s_isGamePaused) UI::PausePanel::Render(window);

    if (INPUTS::s_isDebugVsble) {
        UI::ControlPanel::Render(camera); // 控制面板
        UI::DebugPanel::Render(camera);   // 调试面板
    }
    // 编辑面板
    if (INPUTS::s_isEditMode) UI::EditPanel::Render(camera);
    // 预设库面板
    if (INPUTS::s_isPresetVsble) UI::PresetlibPanel::Render(camera);
}

// 初始化ImGui库
void UIMng::InitImGui() {
    IMGUI_CHECKVERSION(); // 检查ImGui版本
    ImGui::CreateContext(); // 创建ImGui上下文
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // 初始化ImGui的GLFW后端
    ImGui_ImplGlfw_InitForOpenGL(WINDOW::GetWindow(), true);
    // 初始化ImGui的OpenGL3后端，指定着色器版本
    ImGui_ImplOpenGL3_Init("#version 450");
    
}

// 配置ImGui的样式
void UIMng::ConfigureImGuiStyle() {

    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;


    const ImVec4 bg1 = ImVec4(0.196f, 0.196f, 0.196f, 1.0f);        // #323232
    const ImVec4 bg2 = ImVec4(0.118f, 0.118f, 0.118f, 1.0f);        // #1e1e1e
    const ImVec4 bg3 = ImVec4(0.176f, 0.176f, 0.176f, 1.0f);        // #2d2d2d
    const ImVec4 bg4 = ImVec4(0.145f, 0.145f, 0.145f, 1.0f);        // #252525
    
    // 主色调
    const ImVec4 primary = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);      // 蓝色
    const ImVec4 primaryHover = ImVec4(0.36f, 0.69f, 1.00f, 1.00f); // 亮蓝色
    
    // 圆角设置
    style.FrameRounding = 8.0f;
    style.GrabRounding = 8.0f;
    style.WindowRounding = 12.0f;
    style.PopupRounding = 8.0f;
    style.ChildRounding = 8.0f;
    style.ScrollbarRounding = 8.0f;
    style.TabRounding = 8.0f;
    
    // 增加间距和边距
    style.WindowPadding = ImVec2(15, 15);
    style.FramePadding = ImVec2(12, 8);
    style.ItemSpacing = ImVec2(15, 12);
    style.ItemInnerSpacing = ImVec2(10, 8);
    style.WindowBorderSize = 0.0f;
    style.FrameBorderSize = 0.0f;
    
    // 设置颜色
    colors[ImGuiCol_Text] = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg] = bg1;
    colors[ImGuiCol_ChildBg] = bg2;
    colors[ImGuiCol_PopupBg] = bg4;
    colors[ImGuiCol_Border] = ImVec4(0.30f, 0.30f, 0.30f, 0.50f);
    colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg] = bg3;
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.25f, 0.25f, 0.25f, 0.40f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.30f, 0.30f, 0.30f, 0.45f);
    colors[ImGuiCol_TitleBg] = bg3;
    colors[ImGuiCol_TitleBgActive] = bg3;
    colors[ImGuiCol_TitleBgCollapsed] = bg3;
    colors[ImGuiCol_MenuBarBg] = bg3;
    colors[ImGuiCol_ScrollbarBg] = bg4;
    colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.45f, 0.45f, 0.45f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.50f, 0.50f, 0.50f, 0.54f);
    colors[ImGuiCol_CheckMark] = primary;
    colors[ImGuiCol_SliderGrab] = primary;
    colors[ImGuiCol_SliderGrabActive] = primaryHover;
    
    // 按钮颜色
    colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 0.60f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.35f, 0.35f, 0.35f, 0.80f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    
    colors[ImGuiCol_Header] = ImVec4(0.25f, 0.25f, 0.25f, 0.54f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.35f, 0.35f, 0.35f, 0.80f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    
    colors[ImGuiCol_Separator] = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    colors[ImGuiCol_SeparatorHovered] = primary;
    colors[ImGuiCol_SeparatorActive] = primaryHover;
    
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.56f, 0.56f, 0.58f, 0.67f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    
    colors[ImGuiCol_Tab] = ImVec4(0.14f, 0.14f, 0.14f, 0.86f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.25f, 0.25f, 0.25f, 0.80f);
    colors[ImGuiCol_TabActive] = bg3;
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.14f, 0.14f, 0.14f, 0.97f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.20f, 0.20f, 0.20f, 0.60f);
    
    colors[ImGuiCol_DockingPreview] = ImVec4(0.2f, 0.2f, 0.2f, 0.30f);
    colors[ImGuiCol_DockingEmptyBg] = bg2;
}

// 获取窗口中心位置
ImVec2 UIMng::GetWindowCenter(GLFWwindow* window) {
    WINDOW::UpdateWinSize(window); // 更新窗口尺寸

    return ImVec2(WINDOW::GetWidth()/2.0f, WINDOW::GetHeight()/2.0f); // 返回窗口中心位置
}
// 分辨率错误渲染
void UIMng::RenderResolutionError() {
    // 获取窗口尺寸
    int width = WINDOW::GetWidth();
    int height = WINDOW::GetHeight();
    
    // 设置全屏黑色背景
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // 设置文本渲染 (这里使用ImGui作为示例)
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(width, height));
    ImGui::Begin("Resolution Error", nullptr, 
        ImGuiWindowFlags_NoTitleBar | 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBackground
    );
    
    // 居中显示错误信息
    ImVec2 textSize = ImGui::CalcTextSize("该游戏不兼容此分辨率");
    ImVec2 centerPos((width - textSize.x) * 0.5f, (height - textSize.y) * 0.5f);
    
    // 设置红色文本
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
    ImGui::SetCursorPos(centerPos);
    ImGui::Text("该游戏不兼容此分辨率");
    
    // 显示推荐分辨率
    ImVec2 subTextSize = ImGui::CalcTextSize("请使用1280x720或更高分辨率");
    ImVec2 subCenterPos((width - subTextSize.x) * 0.5f, centerPos.y + 30);
    
    ImGui::SetCursorPos(subCenterPos);
    ImGui::Text("请使用1280x720或更高分辨率");
    ImGui::PopStyleColor();
    
    // 显示当前分辨率
    string resText = "当前分辨率: " + std::to_string(width) + "x" + std::to_string(height);
    ImVec2 resTextSize = ImGui::CalcTextSize(resText.c_str());
    ImVec2 resPos((width - resTextSize.x) * 0.5f, subCenterPos.y + 30);
    
    ImGui::SetCursorPos(resPos);
    ImGui::Text("%s", resText.c_str());
    
    ImGui::End();
}
}

#include "kits/file_system.h"
#include "ui/uiMng.h"
#include "core/inputs.h"
#include "core/window.h"
#include "core/monitor.h"
#include "core/time.h"
#include "loaders/fonts.h"

namespace CubeDemo {

// 初始化UI管理器
void UIMng::Init() {
    // 初始化ImGui库
    InitImGui();
    // 配置ImGui的样式
    ConfigureImGuiStyle();
    // 加载自定义字体
    Loaders::Fonts::LoadFonts();
}

// 渲染循环，用于在每一帧中更新和渲染UI
void UIMng::RenderLoop(GLFWwindow* window, Camera camera) {
    HandlePauseMenu(window);    // 处理暂停菜单逻辑

    if (Inputs::s_isDebugVisible) {
        RenderControlPanel(camera); // 渲染控制面板
        RenderDebugPanel(camera);   // 渲染调试面板
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

// 渲染控制面板
void UIMng::RenderControlPanel(Camera& camera) {
    ImGui::Begin("控制面板"); // 开始一个新的ImGui窗口，标题为"Control Panel"
    ImGui::SliderFloat("移动速度", &camera.attribute.movementSpeed, 0.1f, 10.0f); // 添加一个滑动条，用于调整相机移动速度
    
    if (ImGui::Button("全屏")) { // 添加一个按钮，用于切换全屏模式
        Window::ToggleFullscreen(Window::GetWindow());
    }
    ImGui::End(); // 结束ImGui窗口
}

// 处理暂停菜单逻辑
void UIMng::HandlePauseMenu(GLFWwindow* window) {
    if (!Inputs::isGamePaused) return; // 如果游戏未暂停，则直接返回

    const ImVec2 pause_menu_size(400, 350); // 暂停菜单的尺寸
    const ImVec2 window_center = GetWindowCenter(window); // 获取窗口中心位置

    // 使用ImGui的弹出窗口状态管理
    if (ImGui::BeginPopupModal("PauseMenu", nullptr, 
        ImGuiWindowFlags_NoResize |    // 禁止调整窗口大小
        ImGuiWindowFlags_NoMove |      // 禁止移动窗口
        ImGuiWindowFlags_NoCollapse))   // 禁止折叠窗口
    {
        RenderPauseMenuContent(window); // 渲染暂停菜单内容
        ImGui::EndPopup(); // 结束弹出窗口
    } else {
        // 如果弹出窗口未打开，则自动打开
        ImGui::OpenPopup("PauseMenu");
    }

    // 设置窗口属性（只需在首次渲染时设置）
    static bool first_render = true;
    if (first_render) {
        ImGui::SetNextWindowPos(window_center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f)); // 设置窗口位置为窗口中心
        ImGui::SetNextWindowSize(pause_menu_size, ImGuiCond_Always); // 设置窗口大小
        first_render = false;
    }
}

// 获取窗口中心位置
ImVec2 UIMng::GetWindowCenter(GLFWwindow* window) {
    Window::UpdateWinSize(window); // 更新窗口尺寸

    return ImVec2(Window::GetWidth()/2.0f, Window::GetHeight()/2.0f); // 返回窗口中心位置
}

// 渲染暂停菜单内容
void UIMng::RenderPauseMenuContent(GLFWwindow* window) {
    // 居中显示标题
    ImGui::SetCursorPosX((400 - ImGui::CalcTextSize("Game Paused").x) * 0.5f); // 计算标题的水平居中位置
    ImGui::Text("游戏已暂停."); // 显示标题
    ImGui::Separator(); // 添加分隔线

    // 按钮布局
    const ImVec2 button_size(280, 60); // 按钮的尺寸
    if (ImGui::Button("回到游戏", button_size)) { // 添加"回到游戏"按钮
        Inputs::ResumeTheGame(window); // 恢复游戏
        ImGui::CloseCurrentPopup(); // 关闭弹出窗口
    }

    if (ImGui::Button("退出到桌面", button_size)) { // 添加"退出到桌面"按钮
        glfwSetWindowShouldClose(window, true); // 设置窗口关闭标志
    }
}

// 渲染调试面板
void UIMng::RenderDebugPanel(const Camera& camera) {
   const ImGuiWindowFlags window_flags = 
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
    
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.35f); // 半透明背景
    
    if (ImGui::Begin("调试面板", nullptr, window_flags)) {
        // FPS显示
        ImGui::Text("FPS: %d", Time::FPS());
        
        // 摄像机坐标
        const auto& pos = camera.Position;
        ImGui::Text("位置 X: %.1f, Y: %.1f, Z: %.1f)", pos.x, pos.y, pos.z);
        
        // 内存使用
        float memory_usage = Monitor::GetMemoryUsageMB();
        if(memory_usage >= 0) {
            ImGui::Text("内存用量: %.1f MB", memory_usage);
        }
    }
    ImGui::End();
}
}

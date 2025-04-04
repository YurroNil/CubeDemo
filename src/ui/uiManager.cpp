#include "ui/uiManager.h"
#include "core/inputHandler.h"
#include <locale>

// 初始化UI管理器
void UIManager::Init() {
    InitImGui();          // 初始化ImGui库
    ConfigureImGuiStyle(); // 配置ImGui的样式
    LoadFonts();          // 加载自定义字体
}

// 渲染循环，用于在每一帧中更新和渲染UI
void UIManager::RenderLoop(GLFWwindow* window, Camera camera) {
    RenderControlPanel(camera); // 渲染控制面板
    HandlePauseMenu(window);    // 处理暂停菜单逻辑
}

// 初始化ImGui库
void UIManager::InitImGui() {
    IMGUI_CHECKVERSION(); // 检查ImGui版本
    ImGui::CreateContext(); // 创建ImGui上下文
    ImGui_ImplGlfw_InitForOpenGL(WindowManager::GetWindow(), true); // 初始化ImGui的GLFW后端
    ImGui_ImplOpenGL3_Init("#version 330"); // 初始化ImGui的OpenGL3后端，指定着色器版本
}

// 配置ImGui的样式
void UIManager::ConfigureImGuiStyle() {
    ImGuiStyle& style = ImGui::GetStyle(); // 获取ImGui样式对象
    style.WindowPadding = ImVec2(15, 15);    // 设置窗口内边距
    style.FramePadding = ImVec2(10, 10);     // 设置控件内边距
    style.ItemSpacing = ImVec2(10, 15);     // 设置控件之间的间距
    style.ScaleAllSizes(1.5f);              // 放大所有尺寸以适配高DPI屏幕
    ImGui::StyleColorsDark();               // 使用深色主题
}

// 加载自定义字体
void UIManager::LoadFonts() {
    ImGuiIO& io = ImGui::GetIO(); // 获取ImGui的IO对象
    io.Fonts->AddFontFromFileTTF(TextRenderer::Font_Simhei, 30.0f); // 加载中文字体（SimHei），字体大小为30
}

// 渲染控制面板
void UIManager::RenderControlPanel(Camera& camera) {
    ImGui::Begin("Control Panel"); // 开始一个新的ImGui窗口，标题为"Control Panel"
    ImGui::SliderFloat("Movement Speed", &camera.MovementSpeed, 0.1f, 10.0f); // 添加一个滑动条，用于调整相机移动速度
    
    if (ImGui::Button("Toggle Fullscreen")) { // 添加一个按钮，用于切换全屏模式
        WindowManager::ToggleFullscreen(WindowManager::GetWindow());
    }
    ImGui::End(); // 结束ImGui窗口
}

// 处理暂停菜单逻辑
void UIManager::HandlePauseMenu(GLFWwindow* window) {
    if (!InputHandler::isGamePaused) return; // 如果游戏未暂停，则直接返回

    const ImVec2 pauseMenuSize(400, 350); // 暂停菜单的尺寸
    const ImVec2 windowCenter = GetWindowCenter(window); // 获取窗口中心位置

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
    static bool firstRender = true;
    if (firstRender) {
        ImGui::SetNextWindowPos(windowCenter, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f)); // 设置窗口位置为窗口中心
        ImGui::SetNextWindowSize(pauseMenuSize, ImGuiCond_Always); // 设置窗口大小
        firstRender = false;
    }
}

// 获取窗口中心位置
ImVec2 UIManager::GetWindowCenter(GLFWwindow* window) {
    WindowManager::UpdateWindowSize(window); // 更新窗口尺寸

    return ImVec2(WindowManager::s_WindowWidth/2.0f, WindowManager::s_WindowHeight/2.0f); // 返回窗口中心位置
}

// 渲染暂停菜单内容
void UIManager::RenderPauseMenuContent(GLFWwindow* window) {
    // 居中显示标题
    ImGui::SetCursorPosX((400 - ImGui::CalcTextSize("Game Paused").x) * 0.5f); // 计算标题的水平居中位置
    ImGui::Text("GAME PAUSED"); // 显示标题
    ImGui::Separator(); // 添加分隔线

    // 按钮布局
    const ImVec2 buttonSize(280, 60); // 按钮的尺寸
    if (ImGui::Button("Resume Game", buttonSize)) { // 添加"继续游戏"按钮
        InputHandler::ResumeTheGame(window); // 恢复游戏
        ImGui::CloseCurrentPopup(); // 关闭弹出窗口
    }

    if (ImGui::Button("Quit to Desktop", buttonSize)) { // 添加"退出到桌面"按钮
        glfwSetWindowShouldClose(window, true); // 设置窗口关闭标志
    }
}

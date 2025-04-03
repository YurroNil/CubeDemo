#include "ui/uiManager.h"
#include "core/inputHandler.h"
#include <locale>

// 初始化
void UIManager::Init() {
    InitImGui();
    ConfigureImGuiStyle();
    LoadFonts();
}

// 渲染循环
void UIManager::RenderLoop(GLFWwindow* window, Camera camera) {
    RenderControlPanel(camera); // 控制面板
    HandlePauseMenu(window);    // 暂停菜单
}

// 初始化imgui
void UIManager::InitImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(WindowManager::GetWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

// 设置imgui的样式
void UIManager::ConfigureImGuiStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(15, 15);
    style.FramePadding = ImVec2(10, 10);
    style.ItemSpacing = ImVec2(10, 15);
    style.ScaleAllSizes(1.5f);
    ImGui::StyleColorsDark();
}
// 加载字体
void UIManager::LoadFonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF(TextRenderer::Font_Simhei, 30.0f);
}

// 渲染控制面板
void UIManager::RenderControlPanel(Camera& camera) {
    ImGui::Begin("Control Panel");
    ImGui::SliderFloat("Movement Speed", &camera.MovementSpeed, 0.1f, 10.0f);
    
    if (ImGui::Button("Toggle Fullscreen")) {
        WindowManager::ToggleFullscreen(WindowManager::GetWindow());
    }
    ImGui::End();
}

// 暂停菜单
void UIManager::HandlePauseMenu(GLFWwindow* window) {
    if (!InputHandler::isGamePaused) return;

    const ImVec2 pauseMenuSize(400, 350);
    const ImVec2 windowCenter = GetWindowCenter(window);

    // 使用ImGui内置的弹出窗口状态管理
    if (ImGui::BeginPopupModal("PauseMenu", nullptr, 
        ImGuiWindowFlags_NoResize | 
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse))
    {
        RenderPauseMenuContent(window);
        ImGui::EndPopup();
    } else {
        // 自动处理窗口打开逻辑
        ImGui::OpenPopup("PauseMenu");
    }

    // 设置窗口属性（只需在首次渲染时设置）
    static bool firstRender = true;
    if (firstRender) {
        ImGui::SetNextWindowPos(windowCenter, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(pauseMenuSize, ImGuiCond_Always);
        firstRender = false;
    }
}

// 获取窗口中心信息
ImVec2 UIManager::GetWindowCenter(GLFWwindow* window) {
    WindowManager::UpdateWindowSize(window);

    return ImVec2(WindowManager::s_WindowWidth/2.0f, WindowManager::s_WindowHeight/2.0f);
}

// 渲染暂停菜单内容
void UIManager::RenderPauseMenuContent(GLFWwindow* window) {
    // 居中标题
    ImGui::SetCursorPosX((400 - ImGui::CalcTextSize("Game Paused").x) * 0.5f);
    ImGui::Text("GAME PAUSED");
    ImGui::Separator();

    // 按钮布局
    const ImVec2 buttonSize(280, 60);
    if (ImGui::Button("Resume Game", buttonSize)) {
        InputHandler::ResumeTheGame(window);
        ImGui::CloseCurrentPopup();
    }

    if (ImGui::Button("Quit to Desktop", buttonSize)) {
        glfwSetWindowShouldClose(window, true);
    }
}

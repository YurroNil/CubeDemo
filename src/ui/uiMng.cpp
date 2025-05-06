#include "kits/file_system.h"
#include "ui/uiMng.h"
#include "core/inputs.h"
#include "core/window.h"
#include "core/monitor.h"
#include "core/time.h"
namespace CubeDemo {

// 初始化UI管理器
void UIMng::Init() {
    InitImGui();          // 初始化ImGui库
    ConfigureImGuiStyle(); // 配置ImGui的样式
    LoadFonts();          // 加载自定义字体
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

// 加载自定义字体
void UIMng::LoadFonts() {
    if (!fs::exists("../resources/fonts/simhei.ttf")) {
        std::cerr << "字体文件不存在: " << "../resources/fonts/simhei.ttf" << std::endl;
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    
    // 添加中文字符范围
    static const ImWchar ranges[] = {
        // 基础拉丁字符
        0x0020, 0x007F, // 基本ASCII
        0x00A0, 0x00FF, // 拉丁补充

        // 精简中文常用字符集（覆盖99%日常使用）
        0x3000, 0x303F, // 中文标点符号（。，；：「」等）
        0x4E00, 0x62FF, // 常用汉字区1（的、一、是、在等）
        0x6300, 0x77FF, // 常用汉字区2（做、作、使、用等）
        0x7800, 0x8CFF, // 常用汉字区3（器、械、操、控等）
        0x8D00, 0x9FA5, // 常用汉字区4（魔、法、绘、图等）

        0               // 范围终止符
    };

    // 合并加载中文字体
    io.Fonts->AddFontFromFileTTF(
        "../resources/fonts/simhei.ttf",
        30.0f,
        nullptr,
        ranges // 添加字符范围参数
    );
    
    // 合并其他字体（如果有）
    io.Fonts->Build();
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
ImVec2 UIMng::GetWindowCenter(GLFWwindow* window) {
    Window::UpdateWindowSize(window); // 更新窗口尺寸

    return ImVec2(Window::s_WindowWidth/2.0f, Window::s_WindowHeight/2.0f); // 返回窗口中心位置
}

// 渲染暂停菜单内容
void UIMng::RenderPauseMenuContent(GLFWwindow* window) {
    // 居中显示标题
    ImGui::SetCursorPosX((400 - ImGui::CalcTextSize("Game Paused").x) * 0.5f); // 计算标题的水平居中位置
    ImGui::Text("游戏已暂停."); // 显示标题
    ImGui::Separator(); // 添加分隔线

    // 按钮布局
    const ImVec2 buttonSize(280, 60); // 按钮的尺寸
    if (ImGui::Button("回到游戏", buttonSize)) { // 添加"回到游戏"按钮
        Inputs::ResumeTheGame(window); // 恢复游戏
        ImGui::CloseCurrentPopup(); // 关闭弹出窗口
    }

    if (ImGui::Button("退出到桌面", buttonSize)) { // 添加"退出到桌面"按钮
        glfwSetWindowShouldClose(window, true); // 设置窗口关闭标志
    }
}

// 渲染调试面板
void UIMng::RenderDebugPanel(const Camera& camera) {
   const ImGuiWindowFlags windowFlags = 
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
    
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.35f); // 半透明背景
    
    if (ImGui::Begin("调试面板", nullptr, windowFlags)) {
        // FPS显示
        ImGui::Text("FPS: %d", Time::FPS());
        
        // 摄像机坐标
        const auto& pos = camera.Position;
        ImGui::Text("位置 X: %.1f, Y: %.1f, Z: %.1f)", pos.x, pos.y, pos.z);
        
        // 内存使用
        float memUsage = Monitor::GetMemoryUsageMB();
        if(memUsage >= 0) {
            ImGui::Text("内存用量: %.1f MB", memUsage);
        }
    }
    ImGui::End();

}

}
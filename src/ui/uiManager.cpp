// src/ui/uiManager.cpp

#include "ui/uiManager.h"
#include "core/inputHandler.h"

#include <locale>


// 界面相关的初始化
void UIManager::Init() {
    // 初始化ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();   // 配置暗黑主题

    // 绑定GLFW和OpenGL3
    ImGui_ImplGlfw_InitForOpenGL(WindowManager::GetWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 330");  


}



// UI显示（渲染循环内）
void UIManager::RenderLoop(GLFWwindow* window, Camera camera) {

           // 添加滑块控制相机移动速度
        if (ImGui::SliderFloat("Set Movementspeed", &camera.MovementSpeed, 0.1f, 10.0f)) {
            // 值变化时的处理
        }
        
        // 添加按钮触发全屏
        if (ImGui::Button("Toggle Fullscreen")) {
            WindowManager::ToggleFullscreen(WindowManager::GetWindow());
        }

        ImGui::Begin("Main Menu");
        if (ImGui::Button("Button 1")) {
            // 点击开始游戏
        }
        if (ImGui::Button("Button 2")) {
            // 打开设置菜单
        }

        if (ImGui::Button("Quit the Game")) {
            glfwSetWindowShouldClose(window, true);
        }

}


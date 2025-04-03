// include/ui/uiManager.h
#pragma once
#include "graphics/textRenderer.h"

class UIManager {
public:

    static void Init();   // 静态初始化
    static void RenderLoop(GLFWwindow* window, Camera camera);    // 放进渲染循环的主函数

private:
    static void InitImGui();
    static void ConfigureImGuiStyle();
    static void LoadFonts();
    static void RenderControlPanel(Camera& camera);
    static void HandlePauseMenu(GLFWwindow* window);
    static void RenderPauseMenuContent(GLFWwindow* window);
    static ImVec2 GetWindowCenter(GLFWwindow* window);

};

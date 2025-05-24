// include/ui/uiMng.h
#pragma once
#include "core/camera.h"
#include "kits/glfw.h"
#include "kits/imgui.h"

namespace CubeDemo {

class UIMng {
public:

    static void Init();   // 静态初始化
    static void RenderLoop(GLFWwindow* window, Camera camera);    // 放进渲染循环的主函数

    // 渲染调试信息面板
    static void RenderDebugPanel(const Camera& camera);

private:
    static void InitImGui();
    static void ConfigureImGuiStyle();
    static void RenderControlPanel(Camera& camera);
    static void HandlePauseMenu(GLFWwindow* window);
    static void RenderPauseMenuContent(GLFWwindow* window);
    static ImVec2 GetWindowCenter(GLFWwindow* window);

};

}

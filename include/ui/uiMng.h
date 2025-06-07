// include/ui/uiMng.h
#pragma once

namespace CubeDemo {
class Camera;
class UIMng {
public:

    static void Init();   // 静态初始化
    static void RenderLoop(GLFWwindow* window, Camera* camera);    // 放进渲染循环的主函数
    static ImVec2 GetWindowCenter(GLFWwindow* window);

private:
    static void InitImGui();
    static void ConfigureImGuiStyle();
};
}

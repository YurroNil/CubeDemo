// include/managers/uiMng.h
#pragma once
#include "managers/fwd.h"

namespace CubeDemo {
class UIMng {
public:

    static void Init();
    static void RenderInit();
    static void RenderLoop(GLFWwindow* window, Camera* camera);    // 放进渲染循环的主函数
    static ImVec2 GetWindowCenter(GLFWwindow* window);

private:
    static void InitImGui();
    static void ConfigureImGuiStyle();
};
}

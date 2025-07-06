// include/managers/ui/mng.h
#pragma once
#include "managers/fwd.h"

namespace CubeDemo {
class UIMng {
public:
    static void Init();
    static void RenderLoop(GLFWwindow* window, Camera* camera);    // 放进渲染循环的主函数
    static ImVec2 GetWindowCenter(GLFWwindow* window);

    // 添加分辨率错误渲染方法
    static void RenderResolutionError();
    
private:
    static void InitImGui();
    static void ConfigureImGuiStyle();
};
}

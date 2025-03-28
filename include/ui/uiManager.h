// include/ui/uiManager.h
#pragma once
#include "renderer/textRenderer.h"

class UIManager {
public:

    static void Init();   // 静态初始化
    
    static void RenderLoop(GLFWwindow* window, Camera camera);    // 放进渲染循环的主函数
    
private:

};

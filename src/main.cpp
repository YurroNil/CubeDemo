// src/main.cpp
#include "main/init.h"
#include "main/loop.h"
#include "main/cleanup.h"

int main() {
    RL::Init(1);

    /* --------------初始化------------- */

    // 窗口创建
    GLFWwindow* window = CubeDemo::Init();

    // 加载摄像机的指针
    CMR* camera = CMR::GetCamera();

    /* --------------程序主循环------------- */

    CubeDemo::MainLoop(window, camera);
    

    /* --------------资源清理------------- */

    // 程序结束
    CubeDemo::Cleanup(window, camera);
    camera = nullptr;

    return 0;
}

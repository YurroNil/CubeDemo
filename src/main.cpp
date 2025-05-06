// src/main.cpp
#include "init.h"
#include "loop.h"
#include "cleanup.h"

int main() {
    RL::Init(1);
    
    // 程序初始化
    GLFWwindow* window = CubeDemo::Init();
    CMR* camera = CMR::GetCamera(); // 加载摄像机的指针

    // 程序主循环 (包含渲染循环)
    CubeDemo::MainLoop(window, camera);
    
    // 程序结束-资源清理
    CubeDemo::Cleanup(window, camera);
    camera = nullptr;

    return 0;
}

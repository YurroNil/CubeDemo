// src/main.cpp
#include "pch.h"

// 主程序模块
#include "main/init.h"
#include "main/loop.h"
#include "main/cleanup.h"

using CMR = CubeDemo::Camera;

int main(int argc, char* argv[]) {

    /* --------------初始化------------- */

    // 窗口创建
    GLFWwindow* window = CubeDemo::Init(argc, argv);

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

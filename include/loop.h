// include/loop.h

#pragma once
#include "utils/glfwKits.h"
#include "core/camera.h"
#define WIN GLFWwindow* window
#define CAM Camera* camera

namespace CubeDemo {

    // 程序主循环模块
    void MainLoop(WIN, CAM);
    void BeginFrame(CAM);
    void HandleInput(WIN);
    void UpdateModels();
    void HandleWindowSettings(WIN);
    void RenderScene(WIN, CAM);
    void EndFrameHandling(WIN);
    float CalculateAspectRatio(int w, int h);

}
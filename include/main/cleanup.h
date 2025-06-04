// include/main/cleanup.h
#pragma once
#include "main/rendering.h"

namespace CubeDemo {
    class Camera;
    // 程序清理
    void Cleanup(GLFWwindow* window, Camera* camera);

}
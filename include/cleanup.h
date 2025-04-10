// include/cleanup.h

#pragma once

#include "utils/glfwKits.h"
#include "core/camera.h"

namespace CubeDemo {
    // 程序清理
    void Cleanup(GLFWwindow* window, Camera* camera);

}
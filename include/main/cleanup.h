// include/main/cleanup.h

#pragma once

#include "kits/glfw.h"
#include "core/camera.h"

namespace CubeDemo {

    // 程序清理
    void Cleanup(GLFWwindow* window, Camera* camera);

}
// include/init.h

#pragma once
#include <vector>
#include "kits/glfw.h"
#include "mainProgramInc.h"
#include "utils/defines.h"

using RL = CubeDemo::Loaders::Resource;
using CMR = CubeDemo::Camera;

namespace CubeDemo {
    // 程序初始化
    GLFWwindow* Init();
}

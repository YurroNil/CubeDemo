// include/init.h

#pragma once

#include "utils/defines.h"
#include "main/includes.h"

// 乱七八糟的别名
using RL = CubeDemo::Loaders::Resource;
using CMR = CubeDemo::Camera;

namespace CubeDemo {

// 程序初始化
GLFWwindow* Init();

}

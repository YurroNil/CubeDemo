// include/main/init.h
#pragma once
#include "main/rendering.h"
#include "resources/model.h"
#include "loaders/fwd.h"
// 乱七八糟的别名
using RL = CubeDemo::Loaders::Resource;
using CMR = CubeDemo::Camera;
using ModelPtrArray = std::vector<::CubeDemo::Model*>;

namespace CubeDemo {

// 程序初始化
GLFWwindow* Init();

}

// include/init.h

#pragma once
#include "utils/root.h"

// 第三方库
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/string_cast.hpp"
#include "glm/gtx/dual_quaternion.hpp"


// 项目包含的头文件
#include "graphics/mainRenderer.h"
#include "ui/debugInfoManager.h"
#include "core/timeManager.h"
#include "resources/modelManager.h"
#include "graphics/shaderLoader.h"

// 命名空间

namespace CubeDemo {

    // 程序初始化
    GLFWwindow* Init();
}

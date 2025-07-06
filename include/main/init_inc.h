// include/main/init_inc.h
#pragma once
/*
    init因为涉及到多个模块的类，因此需要专门做个inc类整合
*/

// 主程序模块
#include "main/init.h"
#include "main/rendering.h"
// 资源模块
#include "resources/model.h"
// 加载器模块
#include "loaders/model_initer.h"
#include "loaders/progress_tracker.h"
#include "loaders/resource.h"
// 管理器模块
#include "managers/light/mng.h"
#include "managers/model/mng.h"
#include "managers/ui/mng.h"

// 乱七八糟的别名
using RL = CubeDemo::Loaders::Resource;
using CMR = CubeDemo::Camera;
using ModelPtrArray = std::vector<::CubeDemo::Model*>;

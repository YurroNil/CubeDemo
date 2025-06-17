// include/scenes/night_inc.h
#pragma once
/*
    Scenes.Night因为涉及到多个模块的类，因此需要专门做个inc类整合
*/

// 管理器模块
#include "managers/sceneMng.h"
#include "managers/lightMng.h"
#include "managers/modelMng.h"
// 资源模块
#include "resources/model.h"
// 预制体模块
#include "prefabs/shadow_map.h"
// 核心模块
#include "core/camera.h"
#include "core/window.h"
// 工具模块
#include "utils/defines.h"
// 加载器模块
#include "loaders/model_initer.h"

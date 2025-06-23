// include/scenes/default_inc.h
#pragma once
/*
    Scenes.Default因为涉及到多个模块的类，因此需要专门做个inc类整合
*/

// 管理器模块
#include "managers/sceneMng.h"
#include "managers/lightMng.h"
#include "managers/modelMng.h"
// graphics模块
#include "graphics/renderer.h"
// 预制体模块
#include "prefabs/shadow_map.h"
// 工具模块
#include "utils/defines.h"
// 加载器模块
#include "loaders/model_initer.h"

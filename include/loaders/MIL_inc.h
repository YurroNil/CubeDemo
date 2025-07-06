// include/scenes/MIL_inc.h
#pragma once
/*
    model_initer因为涉及到多个模块的类，因此需要专门做个inc类整合
*/

// 加载器模块
#include "loaders/model_initer.h"
#include "loaders/progress_tracker.h"
#include "loaders/model.h"
// 资源模块
#include "resources/model.h"
// 线程模块
#include "threads/task_queue.h"
// 工具模块
#include "utils/defines.h"
// 管理器模块
#include "managers/scene/mng.h"
#include "managers/ui/mng.h"
#include "managers/light/mng.h"
// 界面模块
#include "ui/screens/loading.h"

// include/main/global_variables.h
#pragma once

// 注意: 该头文件必须要在init_inc.h下面包含

namespace CubeDemo {
// 全局变量 (生命周期是到程序结束)
ModelPtrArray MODEL_POINTERS;

// 管理器
SceneMng* SCENE_MNG; LightMng* LIGHT_MNG; ModelMng* MODEL_MNG;

// 暂时采用同步模式
bool DEBUG_ASYNC_MODE = false, RAY_TRACING_ENABLED = false, RT_DEBUG = false;
unsigned int DEBUG_INFO_LV = 0;
}

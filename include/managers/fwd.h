// include/managers/fwd.h
#pragma once

namespace CubeDemo::Managers {

// 向前声明
    // 模型管理器
    class ModelMng; class ModelGetter; class ModelCleanner; class ModelCreater;
    // UI管理器
    class uiMng;
    // 光源管理器
    class LightMng; class LightCreater;
    // 场景管理器
    class SceneMng; class SceneMng; class SceneGetter;
}

using ModelMng = CubeDemo::Managers::ModelMng;
using SceneMng = CubeDemo::Managers::SceneMng;
using LightMng = CubeDemo::Managers::LightMng;
using uiMng = CubeDemo::Managers::uiMng;

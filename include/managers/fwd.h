// include/managers/fwd.h
#pragma once

namespace CubeDemo::Managers {

// 向前声明
    // 模型管理器
    class ModelMng; class ModelGetter; class uiMng;
    class LightMng; class SceneMng;
}

using ModelMng = CubeDemo::Managers::ModelMng;
using SceneMng = CubeDemo::Managers::SceneMng;
using LightMng = CubeDemo::Managers::LightMng;
using uiMng = CubeDemo::Managers::uiMng;

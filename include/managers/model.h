// include/managers/model.h
#pragma once
#include "resources/model.h"
#include "prefabs/lights.h"

namespace CubeDemo::Managers {

class ModelMng {
public:
    static ModelMng* CreateInst();
    static void RemoveInst(ModelMng** ptr);
    void RmvAllModels();
    void RmvAllShaders();

};
}   // namespace CubeDemo::Managers

// 全局别名
using ModelMng = CubeDemo::Managers::ModelMng;

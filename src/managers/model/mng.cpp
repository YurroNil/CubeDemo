// src/managers/model/mng.cpp
#include "pch.h"
#include "managers/model/mng.h"

namespace CubeDemo {
    extern ModelMng* MODEL_MNG; extern unsigned int DEBUG_INFO_LV;
    extern std::vector<Model*> MODEL_POINTERS;
}

namespace CubeDemo::Managers {

// 删除所有着色器以及模型
void ModelMng::RmvAllModels() {
    if(MODEL_POINTERS.empty()) return;
    
    // 删除所有模型
    if(DEBUG_INFO_LV > 1) std::cout << "\n[DELETER]===========" << std::endl;

    for(auto* model : MODEL_POINTERS) {
        if(model == nullptr) continue;
            if(DEBUG_INFO_LV > 0) std::cout << "  删除模型: " << model << " (" << model->GetID() << ")" << std::endl;
            delete model;
    }
    MODEL_POINTERS.clear();
    if(DEBUG_INFO_LV > 1) std::cout << "\n================" << std::endl;
}

// 删除所有着色器(暂不使用)
void ModelMng::RmvAllShaders() {
    for(auto* model : MODEL_POINTERS) {
        if (model == nullptr || model->ModelShader == nullptr) continue;
        delete model->ModelShader;
        model->ModelShader = nullptr;
    }
}
// 创建模型管理器实例
ModelMng* ModelMng::CreateInst() {
    return new ModelMng();
}
// 移除模型管理器实例
void ModelMng::RemoveInst(ModelMng** ptr) {
    delete *ptr; *ptr = nullptr;
}
}   // namespace CubeDemo::Managers
